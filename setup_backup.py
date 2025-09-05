#!/usr/bin/env python3
"""
Son1kVers3 Setup Script
Script de instalación y configuración automática
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Mostrar banner de Son1kVers3"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      🎵 SON1KVERS3 🎵                        ║
    ║                   Resistencia Sonora Setup                    ║
    ║                 Maqueta → Production con IA                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    if sys.version_info < (3, 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} - OK")

def check_system_dependencies():
    """Verificar dependencias del sistema"""
    print("\n🔧 Verificando dependencias del sistema...")
    
    required_packages = {
        'git': 'git --version',
        'docker': 'docker --version',
        'redis': 'redis-server --version'
    }
    
    missing = []
    for package, command in required_packages.items():
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} - OK")
            else:
                missing.append(package)
                print(f"❌ {package} - No encontrado")
        except FileNotFoundError:
            missing.append(package)
            print(f"❌ {package} - No encontrado")
    
    if missing:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing)}")
        print("📋 Instrucciones de instalación:")
        
        if platform.system() == "Darwin":  # macOS
            print("   brew install git docker redis")
        elif platform.system() == "Linux":
            print("   sudo apt update && sudo apt install git docker.io redis-server")
        else:
            print("   Instalar manualmente: git, docker, redis")
        
        return False
    
    return True

def create_virtual_environment():
    """Crear entorno virtual"""
    print("\n🏗️  Configurando entorno virtual...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Entorno virtual ya existe")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Entorno virtual creado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error creando entorno virtual")
        return False

def install_python_dependencies():
    """Instalar dependencias de Python"""
    print("\n📦 Instalando dependencias de Python...")
    
    # Detectar el binario de pip correcto
    if platform.system() == "Windows":
        pip_cmd = ["venv\\Scripts\\pip"]
    else:
        pip_cmd = ["venv/bin/pip"]
    
    try:
        # Actualizar pip
        subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
        
        # Instalar torch primero (es pesado)
        print("🔥 Instalando PyTorch...")
        subprocess.run(pip_cmd + ["install", "torch", "torchvision", "torchaudio"], check=True)
        
        # Instalar dependencias del requirements.txt
        subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_environment_variables():
    """Configurar variables de entorno"""
    print("\n🔐 Configurando variables de entorno...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ Archivo .env ya existe")
        return True
    
    if env_example.exists():
        try:
            # Copiar .env.example a .env
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                content = src.read()
                
                # Generar valores por defecto
                import secrets
                secret_key = secrets.token_urlsafe(32)
                content = content.replace("your-secret-key-here", secret_key)
                
                dst.write(content)
            
            print("✅ Archivo .env creado desde template")
            print("📝 Edita .env para configurar tus API keys y URLs")
            return True
        except Exception as e:
            print(f"❌ Error configurando .env: {e}")
            return False
    else:
        print("❌ Archivo .env.example no encontrado")
        return False

def download_ai_models():
    """Descargar modelos de IA necesarios"""
    print("\n🤖 Preparando modelos de IA...")
    print("📋 Modelos a descargar:")
    print("   - MusicGen (facebook/musicgen-medium) ~2GB")
    print("   - Tortoise TTS para clonación de voz ~1GB")
    
    response = input("¿Descargar modelos ahora? (y/N): ").lower()
    if response == 'y':
        try:
            # Script para pre-descargar modelos
            download_script = """
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

print("Descargando MusicGen...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
print("✅ MusicGen descargado")
            """
            
            # Ejecutar en el entorno virtual
            if platform.system() == "Windows":
                python_cmd = ["venv\\Scripts\\python", "-c", download_script]
            else:
                python_cmd = ["venv/bin/python", "-c", download_script]
            
            subprocess.run(python_cmd, check=True)
            print("✅ Modelos descargados exitosamente")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Error descargando modelos (se pueden descargar después)")
            return True
    else:
        print("⏭️  Modelos se descargarán en primer uso")
        return True

def setup_docker():
    """Configurar servicios Docker"""
    print("\n🐳 Configurando servicios Docker...")
    
    if Path("docker-compose.yml").exists():
        try:
            subprocess.run(["docker-compose", "up", "-d", "redis", "postgres"], check=True)
            print("✅ Servicios Docker iniciados")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Error iniciando Docker (configura manualmente)")
            return False
    else:
        print("❌ docker-compose.yml no encontrado")
        return False

def run_tests():
    """Ejecutar tests para verificar instalación"""
    print("\n🧪 Ejecutando tests de verificación...")
    
    try:
        if platform.system() == "Windows":
            python_cmd = ["venv\\Scripts\\python", "-m", "pytest", "tests/", "-v"]
        else:
            python_cmd = ["venv/bin/python", "-m", "pytest", "tests/", "-v"]
        
        result = subprocess.run(python_cmd, check=True)
        print("✅ Tests pasaron exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Algunos tests fallaron (revisar configuración)")
        return False

def print_next_steps():
    """Mostrar próximos pasos"""
    print("\n" + "="*60)
    print("🎉 ¡SETUP COMPLETADO!")
    print("="*60)
    print("\n📋 PRÓXIMOS PASOS:")
    print("\n1. Activar entorno virtual:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Iniciar servicios:")
    print("   docker-compose up -d")
    
    print("\n3. Iniciar el servidor:")
    print("   python -m uvicorn src.main:app --reload")
    
    print("\n4. Iniciar Celery worker:")
    print("   celery -A src.celery_worker worker --loglevel=info")
    
    print("\n5. Abrir aplicación:")
    print("   http://localhost:8000")
    
    print("\n🎵 ¡Son1kVers3 listo para generar música!")
    print("💡 Para ayuda: python setup.py --help")

def main():
    """Función principal de setup"""
    print_banner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("🎵 Son1kVers3 Setup Script")
        print("\nUso:")
        print("  python setup.py          # Setup completo")
        print("  python setup.py --help   # Mostrar ayuda")
        return
    
    print("🚀 Iniciando configuración automática de Son1kVers3...\n")
    
    steps = [
        ("Verificar Python", check_python_version),
        ("Verificar dependencias sistema", check_system_dependencies),
        ("Crear entorno virtual", create_virtual_environment),
        ("Instalar dependencias Python", install_python_dependencies),
        ("Configurar variables entorno", setup_environment_variables),
        ("Descargar modelos IA", download_ai_models),
        ("Configurar Docker", setup_docker),
        ("Ejecutar tests", run_tests)
    ]
    
    for step_name, step_function in steps:
        print(f"\n{'='*50}")
        print(f"📋 {step_name}")
        print('='*50)
        
        if not step_function():
            print(f"\n❌ Error en: {step_name}")
            print("🛠️  Revisa los errores y ejecuta nuevamente")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()