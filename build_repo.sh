#!/bin/bash

# 🎵 Son1kVers3 - Script para armar repositorio completo
# Ejecutar desde ~/Documents/GitHub/

echo "🎵 Creando repositorio Son1kVers3 completo..."

# 1. Crear directorio principal
echo "📁 Creando estructura de carpetas..."
mkdir -p son1kvers3-completo/{src,tests,frontend/src,docs,assets}
cd son1kvers3-completo

# 2. Crear archivos de configuración básicos
echo "⚙️ Creando archivos de configuración..."

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Son1kVers3 specific
uploads/
output/
models/
*.wav
*.mp3
*.flac
.env
logs/
celerybeat-schedule
dump.rdb

# Node.js (Frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
dist/
build/

# Docker
.dockerignore
EOF

# 3. README básico temporal
cat > README.md << 'EOF'
# 🎵 Son1kVers3 - Resistencia Sonora

**Democratizando la creación musical con IA**

## 🚀 Quick Start

```bash
# Setup automático
python setup.py

# Iniciar aplicación
uvicorn src.main:app --reload
```

## 📋 Funcionalidades

- 🎼 Generación musical con IA (Maqueta → Production)
- 🎤 Clonación de voz expresiva
- 👻 Ghost Studio - Rearreglos creativos
- ⚡ Sistema distribuido con Celery

**Mercado objetivo**: 400+ millones de hispanohablantes
EOF

# 4. Crear __init__.py vacíos para que Python reconozca como paquetes
touch src/__init__.py
touch tests/__init__.py

# 5. requirements.txt básico
cat > requirements.txt << 'EOF'
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0
psycopg2-binary==2.9.9

# Auth
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# AI & Audio
torch>=2.1.0
transformers>=4.35.0
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
scipy>=1.11.0

# Async & Workers
celery[redis]==5.3.4
redis>=5.0.1
aiofiles>=23.2.1

# Utils
python-dotenv>=1.0.0
requests>=2.31.0
pillow>=10.1.0
EOF

echo "✅ Estructura básica creada!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Copiar archivos de tu repositorio actual"
echo "2. Agregar los nuevos archivos de Claude"
echo "3. git init && git add . && git commit"
echo "4. Crear repo en GitHub y hacer push"
echo ""
echo "📁 Directorio creado en: $(pwd)"
EOF