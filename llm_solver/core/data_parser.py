"""
Data file parser for various optimization problem formats.

This module provides utilities to parse and analyze data files for different
optimization problems (TSP, VRP, CSV, etc.) and extract metadata that can be
used by the LLM to generate better BRKGA configurations.
"""

import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DataFormat(Enum):
    """Supported data file formats."""
    TSP = "tsp"
    VRP = "vrp"
    CSV = "csv"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class DataMetadata:
    """Metadata extracted from a data file."""
    format: DataFormat
    file_path: str
    problem_size: int  # Number of cities, items, etc.
    dimension_info: Dict[str, Any]  # Additional dimensions (e.g., capacity, time windows)
    edge_weight_type: Optional[str] = None  # For TSP/VRP
    coordinate_type: Optional[str] = None   # For TSP/VRP
    data_preview: Optional[List[str]] = None  # First few lines for LLM context
    parsed_data: Optional[Any] = None  # Actual parsed data structure

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "format": self.format.value,
            "file_path": self.file_path,
            "problem_size": self.problem_size,
            "dimension_info": self.dimension_info,
            "edge_weight_type": self.edge_weight_type,
            "coordinate_type": self.coordinate_type,
            "data_preview": self.data_preview
        }


class DataFileParser:
    """Parser for various optimization problem data file formats."""

    def __init__(self, max_preview_lines: int = 20):
        """
        Initialize the data file parser.

        Args:
            max_preview_lines: Maximum number of lines to include in preview
        """
        self.max_preview_lines = max_preview_lines

    def detect_format(self, file_path: str) -> DataFormat:
        """
        Detect the format of a data file.

        Args:
            file_path: Path to the data file

        Returns:
            Detected DataFormat
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Check file extension first
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".tsp":
            return DataFormat.TSP
        elif ext == ".vrp":
            return DataFormat.VRP
        elif ext == ".csv":
            return DataFormat.CSV
        elif ext == ".txt":
            # Try to detect based on content
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]

            # Check for TSP/VRP keywords
            content = '\n'.join(first_lines).upper()
            if any(kw in content for kw in ['NAME:', 'TYPE:', 'DIMENSION:', 'EDGE_WEIGHT_TYPE:']):
                if 'VRP' in content or 'CAPACITY:' in content:
                    return DataFormat.VRP
                else:
                    return DataFormat.TSP

            return DataFormat.TXT

        return DataFormat.UNKNOWN

    def parse_file(self, file_path: str, format: Optional[DataFormat] = None) -> DataMetadata:
        """
        Parse a data file and extract metadata.

        Args:
            file_path: Path to the data file
            format: Optional format hint (will auto-detect if None)

        Returns:
            DataMetadata object with extracted information
        """
        if format is None:
            format = self.detect_format(file_path)

        if format == DataFormat.TSP:
            return self._parse_tsp(file_path)
        elif format == DataFormat.VRP:
            return self._parse_vrp(file_path)
        elif format == DataFormat.CSV:
            return self._parse_csv(file_path)
        elif format == DataFormat.TXT:
            return self._parse_txt(file_path)
        else:
            # Treat unknown formats as generic text files
            # This allows the universal solver to accept ANY file format
            # (.ttp, .dat, .json, custom formats, etc.)
            return self._parse_txt(file_path)

    def _parse_tsp(self, file_path: str) -> DataMetadata:
        """
        Parse a TSP (Traveling Salesman Problem) file in TSPLIB format.

        TSPLIB format typically includes:
        - NAME: <name>
        - TYPE: TSP
        - COMMENT: <comment>
        - DIMENSION: <number of cities>
        - EDGE_WEIGHT_TYPE: EUC_2D | GEO | etc.
        - NODE_COORD_SECTION
        - <city_id> <x> <y>
        - EOF
        """
        metadata = {
            "name": None,
            "type": None,
            "comment": None,
            "dimension": 0,
            "edge_weight_type": None
        }

        coordinates = []
        preview_lines = []

        with open(file_path, 'r') as f:
            section = "HEADER"
            line_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Store preview
                if line_count < self.max_preview_lines:
                    preview_lines.append(line)
                    line_count += 1

                # Parse header section
                if section == "HEADER":
                    if line.startswith("NAME"):
                        metadata["name"] = line.split(":", 1)[1].strip()
                    elif line.startswith("TYPE"):
                        metadata["type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("COMMENT"):
                        metadata["comment"] = line.split(":", 1)[1].strip()
                    elif line.startswith("DIMENSION"):
                        metadata["dimension"] = int(line.split(":", 1)[1].strip())
                    elif line.startswith("EDGE_WEIGHT_TYPE"):
                        metadata["edge_weight_type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("NODE_COORD_SECTION"):
                        section = "COORDINATES"

                # Parse coordinate section
                elif section == "COORDINATES":
                    if line.startswith("EOF"):
                        break

                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            city_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            coordinates.append((city_id, x, y))
                        except ValueError:
                            continue

        # Infer dimension if not specified
        if metadata["dimension"] == 0 and coordinates:
            metadata["dimension"] = len(coordinates)

        return DataMetadata(
            format=DataFormat.TSP,
            file_path=file_path,
            problem_size=metadata["dimension"],
            dimension_info={
                "name": metadata["name"],
                "type": metadata["type"],
                "comment": metadata["comment"],
                "num_cities": metadata["dimension"]
            },
            edge_weight_type=metadata["edge_weight_type"],
            coordinate_type="2D" if metadata["edge_weight_type"] in ["EUC_2D", "GEO"] else None,
            data_preview=preview_lines,
            parsed_data={"coordinates": coordinates} if coordinates else None
        )

    def _parse_vrp(self, file_path: str) -> DataMetadata:
        """
        Parse a VRP (Vehicle Routing Problem) file.

        VRP format extends TSP with:
        - CAPACITY: <vehicle capacity>
        - DEMAND_SECTION
        - DEPOT_SECTION
        """
        metadata = {
            "name": None,
            "type": None,
            "dimension": 0,
            "capacity": None,
            "edge_weight_type": None
        }

        coordinates = []
        demands = []
        preview_lines = []

        with open(file_path, 'r') as f:
            section = "HEADER"
            line_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Store preview
                if line_count < self.max_preview_lines:
                    preview_lines.append(line)
                    line_count += 1

                # Parse header section
                if section == "HEADER":
                    if line.startswith("NAME"):
                        metadata["name"] = line.split(":", 1)[1].strip()
                    elif line.startswith("TYPE"):
                        metadata["type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("DIMENSION"):
                        metadata["dimension"] = int(line.split(":", 1)[1].strip())
                    elif line.startswith("CAPACITY"):
                        metadata["capacity"] = int(line.split(":", 1)[1].strip())
                    elif line.startswith("EDGE_WEIGHT_TYPE"):
                        metadata["edge_weight_type"] = line.split(":", 1)[1].strip()
                    elif line.startswith("NODE_COORD_SECTION"):
                        section = "COORDINATES"
                    elif line.startswith("DEMAND_SECTION"):
                        section = "DEMANDS"

                # Parse coordinate section
                elif section == "COORDINATES":
                    if line.startswith("DEMAND_SECTION"):
                        section = "DEMANDS"
                        continue

                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            coordinates.append((node_id, x, y))
                        except ValueError:
                            continue

                # Parse demand section
                elif section == "DEMANDS":
                    if line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
                        break

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            node_id = int(parts[0])
                            demand = int(parts[1])
                            demands.append((node_id, demand))
                        except ValueError:
                            continue

        return DataMetadata(
            format=DataFormat.VRP,
            file_path=file_path,
            problem_size=metadata["dimension"],
            dimension_info={
                "name": metadata["name"],
                "type": metadata["type"],
                "num_nodes": metadata["dimension"],
                "capacity": metadata["capacity"],
                "num_demands": len(demands)
            },
            edge_weight_type=metadata["edge_weight_type"],
            coordinate_type="2D" if metadata["edge_weight_type"] in ["EUC_2D", "GEO"] else None,
            data_preview=preview_lines,
            parsed_data={
                "coordinates": coordinates,
                "demands": demands
            }
        )

    def _parse_csv(self, file_path: str) -> DataMetadata:
        """
        Parse a CSV file.

        Attempts to detect:
        - Number of rows and columns
        - Header presence
        - Data types
        """
        preview_lines = []
        rows = []

        with open(file_path, 'r') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line_count < self.max_preview_lines:
                    preview_lines.append(line)

                rows.append(line.split(','))
                line_count += 1

        num_rows = len(rows)
        num_cols = len(rows[0]) if rows else 0

        # Check if first row is header (contains non-numeric values)
        has_header = False
        if rows:
            try:
                [float(x.strip()) for x in rows[0]]
            except ValueError:
                has_header = True

        # Detect if it's a square matrix (likely distance/time matrix)
        is_square_matrix = (num_rows == num_cols) or (has_header and num_rows - 1 == num_cols)

        return DataMetadata(
            format=DataFormat.CSV,
            file_path=file_path,
            problem_size=num_rows - (1 if has_header else 0),
            dimension_info={
                "num_rows": num_rows,
                "num_cols": num_cols,
                "has_header": has_header,
                "is_square_matrix": is_square_matrix
            },
            data_preview=preview_lines,
            parsed_data={"rows": rows[:100]}  # Store first 100 rows
        )

    def _parse_txt(self, file_path: str) -> DataMetadata:
        """
        Parse a generic text file.

        Attempts to infer structure from the first few lines.
        """
        preview_lines = []
        all_lines = []

        with open(file_path, 'r') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line_count < self.max_preview_lines:
                    preview_lines.append(line)

                all_lines.append(line)
                line_count += 1

        # Try to infer structure
        first_line = all_lines[0] if all_lines else ""
        parts = first_line.split()

        # Check if first line might be problem size
        problem_size = len(all_lines)
        if len(parts) == 1:
            try:
                declared_size = int(parts[0])
                problem_size = declared_size
            except ValueError:
                pass

        return DataMetadata(
            format=DataFormat.TXT,
            file_path=file_path,
            problem_size=problem_size,
            dimension_info={
                "num_lines": len(all_lines),
                "first_line_tokens": len(parts)
            },
            data_preview=preview_lines,
            parsed_data={"lines": all_lines[:100]}
        )

    def generate_llm_context(self, metadata: DataMetadata) -> str:
        """
        Generate a formatted context string for the LLM about the data file.

        Args:
            metadata: DataMetadata object

        Returns:
            Formatted string describing the data file
        """
        context = f"""
DATA FILE INFORMATION:
=====================
File: {os.path.basename(metadata.file_path)}
Format: {metadata.format.value.upper()}
Problem Size: {metadata.problem_size}

"""

        # Add format-specific details
        if metadata.format == DataFormat.TSP:
            context += f"""TSP Details:
- Number of cities: {metadata.dimension_info.get('num_cities', 'Unknown')}
- Edge weight type: {metadata.edge_weight_type or 'Not specified'}
- Coordinate type: {metadata.coordinate_type or 'Not specified'}
"""

        elif metadata.format == DataFormat.VRP:
            context += f"""VRP Details:
- Number of nodes: {metadata.dimension_info.get('num_nodes', 'Unknown')}
- Vehicle capacity: {metadata.dimension_info.get('capacity', 'Not specified')}
- Edge weight type: {metadata.edge_weight_type or 'Not specified'}
"""

        elif metadata.format == DataFormat.CSV:
            context += f"""CSV Details:
- Rows: {metadata.dimension_info.get('num_rows', 'Unknown')}
- Columns: {metadata.dimension_info.get('num_cols', 'Unknown')}
- Has header: {metadata.dimension_info.get('has_header', False)}
- Square matrix: {metadata.dimension_info.get('is_square_matrix', False)}
"""

        # Add data preview
        if metadata.data_preview:
            context += f"\nData Preview (first {len(metadata.data_preview)} lines):\n"
            context += "```\n"
            context += '\n'.join(metadata.data_preview[:10])  # Show max 10 lines
            context += "\n```\n"

        return context


def parse_data_file(file_path: str, format: Optional[DataFormat] = None) -> DataMetadata:
    """
    Convenience function to parse a data file.

    Args:
        file_path: Path to the data file
        format: Optional format hint

    Returns:
        DataMetadata object
    """
    parser = DataFileParser()
    return parser.parse_file(file_path, format)


def generate_data_context(file_path: str) -> str:
    """
    Convenience function to generate LLM context from a data file.

    Args:
        file_path: Path to the data file

    Returns:
        Formatted context string
    """
    parser = DataFileParser()
    metadata = parser.parse_file(file_path)
    return parser.generate_llm_context(metadata)
