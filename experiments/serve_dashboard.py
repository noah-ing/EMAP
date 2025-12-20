#!/usr/bin/env python3
"""
EMAP Dashboard Server

Serves the experiment dashboard and provides live updates via Server-Sent Events (SSE).
Also can run experiments while streaming progress to the dashboard.

Usage:
    # Just serve dashboard (view existing results)
    python experiments/serve_dashboard.py

    # Serve dashboard and run experiments
    python experiments/serve_dashboard.py --run-experiments

    # Open browser automatically
    python experiments/serve_dashboard.py --open
"""

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Any
import time

# Add color support
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print a nice ASCII banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
  ███████╗███╗   ███╗ █████╗ ██████╗
  ██╔════╝████╗ ████║██╔══██╗██╔══██╗
  █████╗  ██╔████╔██║███████║██████╔╝
  ██╔══╝  ██║╚██╔╝██║██╔══██║██╔═══╝
  ███████╗██║ ╚═╝ ██║██║  ██║██║
  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
{Colors.ENDC}
{Colors.GREEN}  Evolution under Multi-Agent Pressure{Colors.ENDC}
{Colors.BLUE}  Dashboard Server{Colors.ENDC}
"""
    print(banner)

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for the dashboard."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        """Custom logging with colors."""
        msg = format % args
        if '200' in msg:
            print(f"  {Colors.GREEN}[OK]{Colors.ENDC} {msg}")
        elif '404' in msg:
            print(f"  {Colors.WARNING}[MISS]{Colors.ENDC} {msg}")
        else:
            print(f"  {Colors.CYAN}[INFO]{Colors.ENDC} {msg}")

    def end_headers(self):
        """Add CORS headers for development."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.path = '/dashboard.html'
        elif self.path == '/api/status':
            self.send_experiment_status()
            return

        super().do_GET()

    def send_experiment_status(self):
        """Send current experiment status as JSON."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        results_dir = Path(self.directory) / 'results'
        status = {}

        budgets = {
            'tight': 2000,
            'medium': 5000,
            'loose': 10000,
            'unconstrained': 50000
        }

        for name, budget in budgets.items():
            result_file = results_dir / f'evolution_budget{budget}_seed42.json'
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        status[name] = json.load(f)
                except Exception as e:
                    status[name] = {'error': str(e)}
            else:
                status[name] = None

        self.wfile.write(json.dumps(status).encode())

def run_server(port: int, directory: str, open_browser: bool = False):
    """Run the HTTP server."""
    handler = partial(DashboardHandler, directory=directory)

    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"

        print(f"\n{Colors.GREEN}{Colors.BOLD}Dashboard server running!{Colors.ENDC}")
        print(f"\n  {Colors.CYAN}URL:{Colors.ENDC} {url}")
        print(f"  {Colors.CYAN}Directory:{Colors.ENDC} {directory}")
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop{Colors.ENDC}\n")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")

def watch_experiments(results_dir: Path):
    """Watch experiment result files and print updates."""
    print(f"\n{Colors.CYAN}Watching for experiment updates...{Colors.ENDC}\n")

    last_modified = {}
    budgets = {
        'Tight (2K)': 2000,
        'Medium (5K)': 5000,
        'Loose (10K)': 10000,
        'Unconstrained (50K)': 50000
    }

    while True:
        for name, budget in budgets.items():
            result_file = results_dir / f'evolution_budget{budget}_seed42.json'

            if result_file.exists():
                mtime = result_file.stat().st_mtime
                if name not in last_modified or mtime > last_modified[name]:
                    last_modified[name] = mtime

                    try:
                        with open(result_file) as f:
                            data = json.load(f)

                        gen = len(data.get('generations', []))
                        total = data.get('config', {}).get('generations', 50)
                        fitness = data.get('best_fitness_per_gen', [0])[-1]
                        agents = data.get('avg_agents_per_gen', [0])[-1]

                        progress = int((gen / total) * 20)
                        bar = '█' * progress + '░' * (20 - progress)

                        print(f"  {Colors.CYAN}{name:20}{Colors.ENDC} "
                              f"[{bar}] "
                              f"Gen {gen:2d}/{total} "
                              f"Fitness: {fitness:.2f} "
                              f"Agents: {agents:.1f}")
                    except Exception as e:
                        pass

        time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="EMAP Dashboard Server")
    parser.add_argument('--port', type=int, default=8080, help='Port to serve on')
    parser.add_argument('--open', action='store_true', help='Open browser automatically')
    parser.add_argument('--watch', action='store_true', help='Watch experiment progress in terminal')
    parser.add_argument('--run-experiments', action='store_true', help='Run experiments while serving')

    args = parser.parse_args()

    print_banner()

    # Determine directory
    experiments_dir = Path(__file__).parent

    if args.watch:
        # Just watch experiments in terminal
        watch_experiments(experiments_dir / 'results')
    else:
        # Start server
        if args.run_experiments:
            # TODO: Integrate with run_experiments.py
            print(f"{Colors.WARNING}Note: --run-experiments not yet implemented{Colors.ENDC}")
            print(f"Run experiments separately with: python experiments/run_experiments.py\n")

        run_server(args.port, str(experiments_dir), args.open)

if __name__ == "__main__":
    main()
