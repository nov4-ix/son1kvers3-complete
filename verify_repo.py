#!/usr/bin/env python3
"""
Son1kVers3 - Script de verificación de repositorio
Verifica que todos los archivos necesarios estén presentes
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Mostrar banner de verificación"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                🔍 SON1KVERS3 REPO VERIFICATION 🔍             ║
    ║            Verificando repositorio completo...               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_file_exists(file_path, required=True):
    """Verificar si un archivo existe"""
    if Path(file_path).exists():
        print(f"✅ {file_path}")
        return True
    else:
        status = "❌" if required else "⚠️ "
        print(f"{status} {file_path} {'(REQUERIDO)' if required else '(OPCIONAL)'}")
        return not required

def check_directory_structure():
    """Verificar estructura de directorios"""
    print("\n📁 VERIFICANDO ESTRUCTURA DE DIRECTORIOS...")
    
    required_dirs = [
        "src",
        "frontend",
        "frontend/src",
        "tests"
    ]
    
    optional_dirs = [
        "docs",
        "assets"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (REQUERIDO)")
            all_good = False
    
    for dir_path in optional_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ (OPCIONAL)")
    
    return all_good

def check_root_files():
    """Verificar archivos en la raíz"""
    print("\n📄 VERIFICANDO ARCHIVOS RAÍZ...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "LICENSE",
        "setup.py"
    ]
    
    optional_files = [
        "pyproject.toml",
        "docker-compose.yml", 
        "Dockerfile",
        ".env.example",
        "SETUP_NIKOLAY.md"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if not check_file_exists(file_path, required=True):
            all_good = False
    
    for file_path in optional_files:
        check_file_exists(file_path, required=False)
    
    return all_good

def check_backend_files():
    """Verificar archivos del backend"""
    print("\n🐍 VERIFICANDO BACKEND (src/)...")
    
    required_files = [
        "src/__init__.py",
        "src/main.py",
        "src/celery_worker.py"
    ]
    
    optional_files = [
        "src/models.py",
        "src/database.py",
        "src/config.py",
        "src/auth.py",
        "src/audio_processing.py",
        "src/voice_expression.py",
        "src/creative_processor.py",
        "src/utils.py",
        "src/monitoring.py"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if not check_file_exists(file_path, required=True):
            all_good = False
    
    for file_path in optional_files:
        check_file_exists(file_path, required=False)
    
    return all_good

def check_frontend_files():
    """Verificar archivos del frontend"""
    print("\n⚛️  VERIFICANDO FRONTEND...")
    
    required_files = [
        "frontend/package.json",
        "frontend/src/App.jsx",
        "frontend/src/main.jsx"
    ]
    
    optional_files = [
        "frontend/src/ErrorBoundary.jsx",
        "frontend/vite.config.js",
        "frontend/index.html"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if not check_file_exists(file_path, required=True):
            all_good = False
    
    for file_path in optional_files:
        check_file_exists(file_path, required=False)
    
    return all_good

def check_tests():
    """Verificar archivos de tests"""
    print("\n🧪 VERIFICANDO TESTS...")
    
    test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
    
    if test_files:
        print(f"✅ Encontrados {len(test_files)} archivos de test:")
        for test_file in test_files:
            print(f"  ✅ {test_file}")
        return True
    else:
        print("⚠️  No se encontraron archivos de test (recomendado crear)")
        return True  # No es crítico

def check_file_contents():
    """Verificar contenido básico de archivos clave"""
    print("\n📋 VERIFICANDO CONTENIDO DE ARCHIVOS CLAVE...")
    
    checks = []
    
    # Verificar setup.py
    if Path("setup.py").exists():
        with open("setup.py", "r") as f:
            content = f.read()
            if "def main():" in content and "Son1kVers3" in content:
                print("✅ setup.py - Contenido válido")
                checks.append(True)
            else:
                print("❌ setup.py - Contenido incompleto")
                checks.append(False)
    
    # Verificar celery_worker.py
    if Path("src/celery_worker.py").exists():
        with open("src/celery_worker.py", "r") as f:
            content = f.read()
            if "celery_app = Celery" in content and "generate_music_task" in content:
                print("✅ celery_worker.py - Contenido válido")
                checks.append(True)
            else:
                print("❌ celery_worker.py - Contenido incompleto")
                checks.append(False)
    
    # Verificar main.py
    if Path("src/main.py").exists():
        with open("src/main.py", "r") as f:
            content = f.read()
            if "FastAPI" in content and "@app.post" in content:
                print("✅ main.py - Contenido válido")
                checks.append(True)
            else:
                print("❌ main.py - Contenido incompleto")
                checks.append(False)
    
    # Verificar requirements.txt
    if Path("requirements.txt").exists():
        with open("requirements.txt", "r") as f:
            content = f.read()
            required_deps = ["fastapi", "celery", "torch", "transformers"]
            missing_deps = [dep for dep in required_deps if dep not in content.lower()]
            
            if not missing_deps:
                print("✅ requirements.txt - Dependencias principales presentes")
                checks.append(True)
            else:
                print(f"❌ requirements.txt - Faltan dependencias: {missing_deps}")
                checks.append(False)
    
    return all(checks) if checks else True

def check_git_status():
    """Verificar estado de Git"""
    print("\n🔗 VERIFICANDO GIT...")
    
    if Path(".git").exists():
        print("✅ Repositorio Git inicializado")
        
        # Verificar si hay commits
        try:
            import subprocess
            result = subprocess.run(["git", "log", "--oneline", "-1"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Commits presentes")
                print(f"  Último commit: {result.stdout.strip()}")
            else:
                print("⚠️  No hay commits aún (ejecutar git commit)")
        except:
            print("⚠️  No se puede verificar commits")
        
        # Verificar remote
        try:
            result = subprocess.run(["git", "remote", "-v"], 
                                  capture_output=True, text=True)
            if "origin" in result.stdout:
                print("✅ Remote origin configurado")
            else:
                print("⚠️  Remote origin no configurado")
        except:
            print("⚠️  No se puede verificar remote")
        
        return True
    else:
        print("❌ Git no inicializado (ejecutar git init)")
        return False

def generate_report():
    """Generar reporte final"""
    print("\n" + "="*60)
    print("📊 REPORTE DE VERIFICACIÓN")
    print("="*60)
    
    all_checks = [
        ("Estructura directorios", check_directory_structure()),
        ("Archivos raíz", check_root_files()),
        ("Backend Python", check_backend_files()),
        ("Frontend React", check_frontend_files()),
        ("Tests", check_tests()),
        ("Contenido archivos", check_file_contents()),
        ("Git setup", check_git_status())
    ]
    
    passed = sum([1 for _, status in all_checks if status])
    total = len(all_checks)
    
    print(f"\n📈 RESULTADOS: {passed}/{total} verificaciones pasadas")
    
    if passed == total:
        print("\n🎉 ¡REPOSITORIO COMPLETO Y LISTO!")
        print("✅ Todos los archivos necesarios presentes")
        print("✅ Contenido de archivos válido")
        print("✅ Git configurado correctamente")
        print("\n🚀 PRÓXIMOS PASOS:")
        print("1. git add . && git commit -m 'Repositorio completo'")
        print("2. Crear repo en GitHub")
        print("3. git remote add origin [URL]")
        print("4. git push -u origin main")
        print("5. Enviar a Nikolay")
    elif passed >= total - 2:
        print("\n⚠️  REPOSITORIO CASI LISTO")
        print("✅ Estructura principal correcta")
        print("🔧 Pequeños ajustes necesarios")
        print("\n📋 COMPLETAR:")
        for name, status in all_checks:
            if not status:
                print(f"  ❌ {name}")
    else:
        print("\n🚨 REPOSITORIO INCOMPLETO")
        print("❌ Faltan archivos importantes")
        print("\n📋 PENDIENTES:")
        for name, status in all_checks:
            if not status:
                print(f"  ❌ {name}")
    
    print(f"\n🎵 Son1kVers3 - Resistencia Sonora")
    print("💡 Para ayuda: python verify_repo.py --help")

def main():
    """Función principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("🔍 Son1kVers3 Repository Verification")
        print("\nUso:")
        print("  python verify_repo.py          # Verificación completa")
        print("  python verify_repo.py --help   # Mostrar ayuda")
        return
    
    print_banner()
    print("🚀 Iniciando verificación de repositorio Son1kVers3...\n")
    
    generate_report()

if __name__ == "__main__":
    main()