#!/bin/bash
# =============================================================================
# Oracle Cloud Always Free — Setup per Nowcast Cardedeu
# Executar UN SOL COP a la VM ARM d'Oracle Cloud
# =============================================================================
#
# Pre-requisits:
#   1. VM creada a Oracle Cloud (Always Free ARM Ampere A1, 1 OCPU, 6 GB RAM)
#      - Imatge: Oracle Linux 9 o Ubuntu 22.04 minimal
#      - Boot volume: 47 GB (Always Free limit)
#      - SSH key configurada
#   2. SSH connectat a la VM: ssh -i <clau> opc@<ip-publica>
#
# Ús:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================
set -euo pipefail

echo "========================================="
echo "🌧️ Nowcast Cardedeu — Oracle Cloud Setup"
echo "========================================="

# ── Detectar distro ──
if command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
    echo "📦 Detectat Oracle Linux / RHEL (dnf)"
elif command -v apt-get &>/dev/null; then
    PKG_MANAGER="apt"
    echo "📦 Detectat Ubuntu / Debian (apt)"
else
    echo "❌ Gestor de paquets no reconegut. Instal·la manualment Python 3.12 i git."
    exit 1
fi

# ── Instal·lar dependències del sistema ──
echo ""
echo "📦 Instal·lant dependències del sistema..."
if [ "$PKG_MANAGER" = "dnf" ]; then
    sudo dnf install -y python3.12 python3.12-pip git
elif [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git
fi

# ── Clonar el repositori ──
REPO_DIR="$HOME/nowcast-cardedeu"
if [ -d "$REPO_DIR" ]; then
    echo "📂 Repositori ja existeix a $REPO_DIR, fent git pull..."
    cd "$REPO_DIR" && git pull
else
    echo "📂 Clonant repositori..."
    git clone https://github.com/albertolive/nowcast-cardedeu.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Crear venv i instal·lar dependències ──
echo ""
echo "🐍 Creant entorn virtual Python 3.12..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependències instal·lades correctament."

# ── Configurar variables d'entorn ──
ENV_FILE="$REPO_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "🔐 Creant fitxer .env per a secrets..."
    echo "   Hauràs d'editar-lo amb els teus valors reals:"
    echo "   nano $ENV_FILE"
    cat > "$ENV_FILE" << 'ENVEOF'
# Secrets per Nowcast Cardedeu
# Edita amb els teus valors reals: nano ~/.env-nowcast
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
METEOCAT_API_KEY=
AEMET_API_KEY=
ENVEOF
    chmod 600 "$ENV_FILE"
    echo "⚠️  IMPORTANT: Edita $ENV_FILE amb els teus secrets!"
else
    echo "🔐 Fitxer .env ja existeix, no el sobreescriu."
fi

# ── Configurar git per push automàtic ──
echo ""
echo "🔧 Configurant git..."
cd "$REPO_DIR"
git config user.name "nowcast-oracle-vm"
git config user.email "nowcast-oracle-vm@users.noreply.github.com"

# ── Instal·lar servei systemd ──
echo ""
echo "⏱️  Instal·lant servei systemd (timer cada 10 min)..."
sudo cp "$REPO_DIR/oracle-cloud/nowcast-predict.service" /etc/systemd/system/
sudo cp "$REPO_DIR/oracle-cloud/nowcast-predict.timer" /etc/systemd/system/

# Substituir l'usuari i path al servei
CURRENT_USER=$(whoami)
sudo sed -i "s|__USER__|$CURRENT_USER|g" /etc/systemd/system/nowcast-predict.service
sudo sed -i "s|__REPO_DIR__|$REPO_DIR|g" /etc/systemd/system/nowcast-predict.service

sudo systemctl daemon-reload
sudo systemctl enable nowcast-predict.timer
sudo systemctl start nowcast-predict.timer

echo ""
echo "========================================="
echo "✅ Setup complet!"
echo "========================================="
echo ""
echo "Passos pendents:"
echo "  1. Editar secrets:    nano $ENV_FILE"
echo "  2. Configurar git push (triar UNA opció):"
echo "     a) SSH key: ssh-keygen -t ed25519 && cat ~/.ssh/id_ed25519.pub"
echo "        (afegir la clau pública a GitHub → Settings → SSH keys)"
echo "        git remote set-url origin git@github.com:albertolive/nowcast-cardedeu.git"
echo "     b) Token: git remote set-url origin https://<GITHUB_TOKEN>@github.com/albertolive/nowcast-cardedeu.git"
echo "  3. Verificar timer:   systemctl status nowcast-predict.timer"
echo "  4. Executar prova:    sudo systemctl start nowcast-predict.service"
echo "  5. Veure logs:        journalctl -u nowcast-predict.service -f"
echo ""
echo "El timer s'executarà cada 10 min entre 6:00-23:00 Barcelona."
echo "Fora d'horari, el servei salta l'execució automàticament."
