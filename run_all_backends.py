#!/usr/bin/env python3
"""
Run All AI Project Backends
This script starts all backend services for the AI portfolio projects
"""

import subprocess
import threading
import time
import sys
import os
from datetime import datetime

class BackendManager:
    def __init__(self):
        self.backends = {
            'FireWise AI': {
                'file': 'firewise_ai_backend.py',
                'port': 5000,
                'description': 'Forest Fire Detection System'
            },
            'Breast Cancer Detection': {
                'file': 'breast_cancer_backend.py',
                'port': 5001,
                'description': 'Medical AI for Cancer Detection'
            },
            'Traffic Prediction': {
                'file': 'traffic_prediction_backend.py',
                'port': 5002,
                'description': 'Smart Traffic Management System'
            },
            'WhatsApp Chatbot': {
                'file': 'whatsapp_chatbot_backend.py',
                'port': 5003,
                'description': 'Intelligent WhatsApp Business Bot'
            },
            'Reviews System': {
                'file': 'reviews_backend.py',
                'port': 5004,
                'description': 'Customer Reviews Management System'
            }
        }
        self.processes = {}
        self.running = True
    
    def print_banner(self):
        """Print startup banner"""
        print("=" * 80)
        print("🚀 AI PORTFOLIO BACKEND MANAGER")
        print("=" * 80)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Python version: {sys.version}")
        print(f"📁 Working directory: {os.getcwd()}")
        print("=" * 80)
        print()
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'flask', 'tensorflow', 'opencv-python', 
            'scikit-learn', 'numpy', 'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("📦 Install with: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies are installed!")
        return True
    
    def start_backend(self, name, config):
        """Start a single backend service"""
        try:
            print(f"🚀 Starting {name}...")
            print(f"   📄 File: {config['file']}")
            print(f"   🌐 Port: {config['port']}")
            print(f"   📝 Description: {config['description']}")
            
            # Start the process
            process = subprocess.Popen([
                sys.executable, config['file']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes[name] = process
            
            # Wait a moment to check if it started successfully
            time.sleep(2)
            
            if process.poll() is None:
                print(f"   ✅ {name} started successfully on port {config['port']}")
                print(f"   🔗 URL: http://localhost:{config['port']}")
            else:
                stdout, stderr = process.communicate()
                print(f"   ❌ {name} failed to start")
                print(f"   Error: {stderr}")
                
        except Exception as e:
            print(f"   ❌ Error starting {name}: {e}")
    
    def start_all_backends(self):
        """Start all backend services"""
        print("🚀 Starting all backend services...\n")
        
        threads = []
        for name, config in self.backends.items():
            if os.path.exists(config['file']):
                thread = threading.Thread(
                    target=self.start_backend, 
                    args=(name, config)
                )
                thread.start()
                threads.append(thread)
                time.sleep(1)  # Stagger startup
            else:
                print(f"⚠️  {config['file']} not found, skipping {name}")
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print("\n" + "=" * 80)
        print("🎉 BACKEND STARTUP COMPLETE")
        print("=" * 80)
        
        # Show running services
        self.show_status()
    
    def show_status(self):
        """Show status of all services"""
        print("\n📊 SERVICE STATUS:")
        print("-" * 50)
        
        for name, config in self.backends.items():
            if name in self.processes:
                process = self.processes[name]
                if process.poll() is None:
                    status = "🟢 RUNNING"
                    url = f"http://localhost:{config['port']}"
                else:
                    status = "🔴 STOPPED"
                    url = "N/A"
            else:
                status = "⚪ NOT STARTED"
                url = "N/A"
            
            print(f"{name:25} {status:12} {url}")
        
        print("-" * 50)
        print("\n💡 USAGE TIPS:")
        print("• Test APIs using curl or Postman")
        print("• Check logs for debugging information")
        print("• Press Ctrl+C to stop all services")
        print()
    
    def monitor_services(self):
        """Monitor running services"""
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        print(f"⚠️  {name} has stopped unexpectedly")
                        # Could implement auto-restart here
                        
        except KeyboardInterrupt:
            self.stop_all_services()
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        self.running = False
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:
                    print(f"   Stopping {name}...")
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"   ✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"   🔥 Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"   ❌ Error stopping {name}: {e}")
        
        print("🏁 All services stopped")
        sys.exit(0)
    
    def run(self):
        """Main run method"""
        self.print_banner()
        
        if not self.check_dependencies():
            print("\n❌ Please install missing dependencies first")
            sys.exit(1)
        
        print()
        self.start_all_backends()
        
        try:
            self.monitor_services()
        except KeyboardInterrupt:
            self.stop_all_services()

def main():
    """Main function"""
    manager = BackendManager()
    manager.run()

if __name__ == '__main__':
    main()