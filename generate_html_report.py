  #!/usr/bin/env python3
"""
Generate HTML report with embedded SVG charts from BRKGA results.

Usage:
    python generate_html_report.py ./berlin52/
    python generate_html_report.py --dir ./berlin52/ -o report.html
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime


def svg_convergence(convergence, width=600, height=300):
    if not convergence or len(convergence) < 2:
        return '<text x="50%" y="50%" text-anchor="middle">No convergence data</text>'
   
    gens, fitness = zip(*convergence)
    min_gen, max_gen = min(gens), max(gens)
    min_fit, max_fit = min(fitness), max(fitness)
   
    fit_range = max_fit - min_fit if max_fit > min_fit else 1
    min_fit -= fit_range * 0.05
    max_fit += fit_range * 0.05
    fit_range = max_fit - min_fit
   
    gen_range = max_gen - min_gen if max_gen > min_gen else 1
   
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
   
    def x(g):
        return margin + (g - min_gen) / gen_range * plot_w
   
    def y(f):
        return margin + (1 - (f - min_fit) / fit_range) * plot_h
   
    points = ' '.join(f"{x(g):.1f},{y(f):.1f}" for g, f in convergence)
    best_idx = fitness.index(min(fitness))
    best_gen, best_fit = gens[best_idx], fitness[best_idx]
   
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <g stroke="#ddd" stroke-width="1">
    <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}"/>
    <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}"/>
  </g>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12">Generation</text>
  <text x="15" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90,15,{height/2})">Makespan</text>
  <text x="{margin-5}" y="{margin+5}" text-anchor="end" font-size="10">{max_fit:.0f}</text>
  <text x="{margin-5}" y="{height-margin+5}" text-anchor="end" font-size="10">{min_fit:.0f}</text>
  <text x="{margin}" y="{height-margin+15}" text-anchor="middle" font-size="10">{min_gen}</text>
  <text x="{width-margin}" y="{height-margin+15}" text-anchor="middle" font-size="10">{max_gen}</text>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <circle cx="{x(best_gen):.1f}" cy="{y(best_fit):.1f}" r="5" fill="red"/>
  <text x="{x(best_gen)+10:.1f}" y="{y(best_fit)-5:.1f}" font-size="10" fill="red">Best: {best_fit:.1f} @ gen {best_gen}</text>
  <text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Convergence</text>
</svg>'''


def svg_gantt(routes, width=1000, height=None):
    if not routes:
        return '<text x="50%" y="50%" text-anchor="middle">No route data</text>'
   
    n_agents = len(routes)
    makespan = max(stop['time'] for r in routes for stop in r['stops'])
    if makespan == 0:
        makespan = 1
   
    margin_left = 70
    margin_right = 30
    margin_top = 60
    margin_bottom = 50
    row_height = 90
    radius = 14
   
    if height is None:
        height = margin_top + margin_bottom + n_agents * row_height
   
    plot_w = width - margin_left - margin_right
   
    def x(t):
        return margin_left + t / makespan * plot_w
   
    def y(a):
        return margin_top + a * row_height + row_height / 2
   
    customer_events = {}
    for agent_data in routes:
        agent = agent_data['agent']
        for stop in agent_data['stops']:
            time, op, node = stop['time'], stop['op'], stop['node']
            if node == 0:
                continue
            if node not in customer_events:
                customer_events[node] = {}
            customer_events[node][op] = {'agent': agent, 'time': time, 'x': x(time), 'y': y(agent)}
   
    background_elements = []
    cross_agent_lines = []
    same_agent_lines = []
    circle_elements = []
    same_agent_bump_counter = {}
   
    for node, events in sorted(customer_events.items()):
        if 'P' not in events or 'D' not in events:
            continue
        pick, drop = events['P'], events['D']
        px, py, dx, dy = pick['x'], pick['y'], drop['x'], drop['y']
       
        if pick['agent'] != drop['agent']:
            cross_agent_lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{dx:.1f}" y2="{dy:.1f}" stroke="#f97316" stroke-width="2.5" stroke-opacity="0.7" stroke-dasharray="8,4"/>')
        else:
            agent_key = pick['agent']
            bump_index = same_agent_bump_counter.get(agent_key, 0)
            same_agent_bump_counter[agent_key] = bump_index + 1
            bump_offsets = [22, 30, 38, 26, 34, 42]
            bump_y = py - bump_offsets[bump_index % len(bump_offsets)]
            path = f'M {px:.1f} {py - radius:.1f} L {px:.1f} {bump_y:.1f} L {dx:.1f} {bump_y:.1f} L {dx:.1f} {dy - radius:.1f}'
            same_agent_lines.append(f'<path d="{path}" fill="none" stroke="#8b5cf6" stroke-width="1.5" stroke-opacity="0.6"/>')
   
    for agent_data in routes:
        agent = agent_data['agent']
        agent_y = y(agent)
        background_elements.append(f'<rect x="{margin_left}" y="{agent_y - 22:.1f}" width="{plot_w}" height="44" fill="#f3f4f6" rx="4"/>')
        background_elements.append(f'<text x="{margin_left-10}" y="{agent_y+5:.1f}" text-anchor="end" font-size="13" font-weight="600" fill="#374151">Agent {agent}</text>')
       
        for stop in agent_data['stops']:
            time, op, node = stop['time'], stop['op'], stop['node']
            cx, cy = x(time), agent_y
            if node == 0:
                circle_elements.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#9ca3af" stroke="#6b7280" stroke-width="1.5"/>')
            elif op == 'D':
                circle_elements.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="#3b82f6" stroke="#1e40af" stroke-width="2"/>')
                circle_elements.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" font-size="11" font-weight="bold" fill="white" text-anchor="middle">{node}</text>')
            elif op == 'P':
                circle_elements.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="#22c55e" stroke="#15803d" stroke-width="2"/>')
                circle_elements.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" font-size="11" font-weight="bold" fill="white" text-anchor="middle">{node}</text>')
   
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">Agent Timeline</text>
  <g transform="translate({width-280}, 8)">
    <circle cx="10" cy="14" r="9" fill="#22c55e" stroke="#15803d" stroke-width="1.5"/>
    <text x="26" y="18" font-size="12" fill="#374151">Pick</text>
    <circle cx="70" cy="14" r="9" fill="#3b82f6" stroke="#1e40af" stroke-width="1.5"/>
    <text x="86" y="18" font-size="12" fill="#374151">Drop</text>
    <line x1="130" y1="14" x2="160" y2="14" stroke="#f97316" stroke-width="2.5" stroke-dasharray="6,3"/>
    <text x="166" y="18" font-size="12" fill="#374151">X-agent</text>
  </g>
  <g>{''.join(background_elements)}</g>
  <g>{''.join(cross_agent_lines)}</g>
  <g>{''.join(same_agent_lines)}</g>
  <g>{''.join(circle_elements)}</g>
  <line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#374151" stroke-width="2"/>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13" fill="#374151">Time</text>
  <text x="{margin_left}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="11" fill="#6b7280">0</text>
  <text x="{width-margin_right}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="11" fill="#6b7280">{makespan:.0f}</text>
</svg>'''
    return svg


def svg_comparison_chart(results, width=700, height=400):
    if not results:
        return ''
   
    margin_left, margin_right, margin_top, margin_bottom = 120, 50, 50, 40
    n = len(results)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_h = min(40, plot_h / n * 0.7)
   
    makespans = [r['makespan'] for r in results]
    min_m, max_m = min(makespans) * 0.95, max(makespans) * 1.02
    range_m = max_m - min_m if max_m > min_m else 1
   
    def x(m):
        return margin_left + (m - min_m) / range_m * plot_w
    def y(i):
        return margin_top + i * (plot_h / n) + (plot_h / n - bar_h) / 2
   
    colors = ['#22c55e', '#3b82f6', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4']
    bars = []
   
    for i, r in enumerate(results):
        m = r['makespan']
        color = colors[i % len(colors)]
        bar_width = x(m) - margin_left
        bars.append(f'<rect x="{margin_left}" y="{y(i):.1f}" width="{bar_width:.1f}" height="{bar_h}" fill="{color}" opacity="0.8" rx="4"/>')
        bars.append(f'<text x="{margin_left-5}" y="{y(i)+bar_h/2+5:.1f}" text-anchor="end" font-size="12" font-weight="500">{r["dir"]}</text>')
        bars.append(f'<text x="{x(m)+5:.1f}" y="{y(i)+bar_h/2+5:.1f}" font-size="12" font-weight="bold">{m:.1f}</text>')
   
    best_idx = makespans.index(min(makespans))
    bars.append(f'<text x="{x(results[best_idx]["makespan"])+50:.1f}" y="{y(best_idx)+bar_h/2+5:.1f}" font-size="14" fill="#22c55e" font-weight="bold">â˜… BEST</text>')
   
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">Configuration Comparison</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#374151" stroke-width="2"/>
  <line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#374151" stroke-width="2"/>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12" fill="#6b7280">Makespan</text>
  {''.join(bars)}
</svg>'''


def find_json_files(paths):
    json_files = []
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            json_files.extend(glob.glob(os.path.join(path, 'soln*.json')))
            json_files.extend(glob.glob(os.path.join(path, '*', 'soln*.json')))
            json_files.extend(glob.glob(os.path.join(path, '*', '*', 'soln*.json')))
        elif '*' in path:
            json_files.extend(glob.glob(path))
        elif path.endswith('.json') and os.path.exists(path):
            json_files.append(path)
    return sorted(set(json_files))


def generate_report(json_files, output_path):
    results = []
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            dir_name = os.path.basename(os.path.dirname(json_path))
            results.append({
                'path': json_path, 'dir': dir_name,
                'makespan': data.get('makespan'), 'solve_time': data.get('solve_time_seconds'),
                'routes': data.get('routes', []), 'convergence': data.get('convergence', []),
                'problem': data.get('problem', {}),
            })
            print(f"  Loaded: {dir_name} -> makespan={data.get('makespan'):.2f}")
        except Exception as e:
            print(f"  Error loading {json_path}: {e}")
   
    if not results:
        print("No results to report!")
        return
   
    results.sort(key=lambda r: r['makespan'] or float('inf'))
    best_makespan = results[0]['makespan']
   
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BRKGA Suite Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f8fafc; }}
        h1 {{ color: #1e40af; margin-bottom: 5px; }}
        h2 {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
        .meta {{ color: #6b7280; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 12px; text-align: left; }}
        th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
        tr:nth-child(even) {{ background: #f9fafb; }}
        .best {{ background: #dcfce7 !important; font-weight: bold; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .charts {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .chart {{ background: white; padding: 10px; border-radius: 8px; border: 1px solid #e5e7eb; }}
        svg {{ max-width: 100%; height: auto; }}
        .legend-note {{ font-size: 12px; color: #6b7280; margin-top: 10px; padding: 10px; background: #f8fafc; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>ðŸ§¬ BRKGA Suite Results</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Configurations: {len(results)} | Best: {best_makespan:.2f}</p>
   
    <div class="section">
        <h2>ðŸ“Š Comparison</h2>
        <div class="charts"><div class="chart">{svg_comparison_chart(results)}</div></div>
    </div>
   
    <div class="section">
        <h2>ðŸ“‹ Summary Table</h2>
        <table>
            <tr><th>Rank</th><th>Configuration</th><th>Makespan</th><th>vs Best</th><th>Time (s)</th><th>Customers</th><th>Agents</th></tr>
'''
   
    for i, r in enumerate(results):
        row_class = 'best' if i == 0 else ''
        makespan_str = f"{r['makespan']:.2f}" if r['makespan'] else 'N/A'
        time_str = f"{r['solve_time']:.1f}" if r['solve_time'] else 'N/A'
        vs_best = "â˜… BEST" if i == 0 else f"+{((r['makespan'] - best_makespan) / best_makespan) * 100:.1f}%"
        html += f'<tr class="{row_class}"><td>{i+1}</td><td>{r["dir"]}</td><td>{makespan_str}</td><td>{vs_best}</td><td>{time_str}</td><td>{r["problem"].get("n_customers", "N/A")}</td><td>{r["problem"].get("n_agents", "N/A")}</td></tr>\n'
   
    html += '</table></div>\n'
   
    for i, r in enumerate(results):
        makespan_str = f"{r['makespan']:.2f}" if r['makespan'] else 'N/A'
        badge = "ðŸ †" if i == 0 else f"#{i+1}"
        html += f'''
    <div class="section">
        <h2>{badge} {r['dir']} (Makespan: {makespan_str})</h2>
        <div class="charts">
            <div class="chart">{svg_convergence(r['convergence'])}</div>
            <div class="chart">{svg_gantt(r['routes']) if r['routes'] else '<p>No route data</p>'}</div>
        </div>
        <div class="legend-note"><strong>Legend:</strong> ðŸŸ¢ Green = Pickup | ðŸ”µ Blue = Dropoff | <span style="color:#f97316">â” â” </span> Orange dashed = Cross-agent</div>
    </div>
'''
   
    html += '</body></html>'
   
    with open(output_path, 'w') as f:
        f.write(html)
   
    print(f"\nâœ… Report saved to: {output_path}")
    print(f"   Best configuration: {results[0]['dir']} with makespan {results[0]['makespan']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from BRKGA results')
    parser.add_argument('paths', nargs='*', help='Directories or JSON files')
    parser.add_argument('--dir', '-d', help='Results directory')
    parser.add_argument('-o', '--output', default='brkga_report.html', help='Output HTML file')
    args = parser.parse_args()
   
    paths = args.paths or []
    if args.dir:
        paths.append(args.dir)
   
    if not paths:
        print("Usage: python generate_html_report.py ./berlin52/")
        sys.exit(1)
   
    print(f"Searching for JSON files in: {paths}")
    json_files = find_json_files(paths)
   
    if not json_files:
        print("No JSON files found!")
        sys.exit(1)
   
    print(f"Found {len(json_files)} JSON files:")
    generate_report(json_files, args.output)


if __name__ == '__main__':
    main()