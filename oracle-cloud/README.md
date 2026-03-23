# Oracle Cloud Always Free — Nowcast Cardedeu

Migra la predicció (`predict_now.py`) de GitHub Actions a una VM gratuïta d'Oracle Cloud, estalviant ~1.100 minuts/mes d'Actions.

## Per què Oracle Cloud?

| Aspecte | GitHub Actions | Oracle Cloud VM |
|---------|---------------|-----------------|
| **Cost** | ~1.100 min/mes (54 min/dia) | $0 per sempre (Always Free) |
| **Interval** | 10 min (via cron-job.org) | 10 min (systemd timer) |
| **Cold start** | ~30s (checkout + venv cache) | 0s (tot instal·lat) |
| **Fiabilitat** | Depèn de cron-job.org + GitHub queue | Timer local, sense dependències |
| **Persistència** | Necessita git push cada execució | Fitxers locals + git push |
| **Concurrència** | Risc de push conflicts | Un sol procés, sense conflictes |

## Always Free Tier (verificat març 2026)

- **ARM Ampere A1**: fins a 4 OCPU + 24 GB RAM (3.000 OCPU-hours/mes)
- **Boot volume**: fins a 200 GB
- **Xarxa**: 10 TB egress/mes
- **Sense caducitat**: no és un trial, és permanent
- Font: [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)

Per al nostre cas, 1 OCPU + 6 GB RAM és molt més del necessari.

## Fitxers

```
oracle-cloud/
├── setup.sh                    # Script d'instal·lació (executar 1 cop)
├── run-predict.sh              # Script que executa la predicció
├── nowcast-predict.service     # Unitat systemd (oneshot)
├── nowcast-predict.timer       # Timer systemd (cada 10 min)
└── README.md                   # Aquesta documentació
```

## Instal·lació pas a pas

### 1. Crear la VM a Oracle Cloud

1. Registra't a [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)
2. Crea una instància Compute:
   - **Imatge**: Oracle Linux 9 o Ubuntu 22.04 minimal
   - **Shape**: VM.Standard.A1.Flex (1 OCPU, 6 GB RAM)
   - **Boot volume**: 47 GB (dins del límit Always Free)
   - **SSH key**: afegeix la teva clau pública
3. Configura la **Security List** per permetre trànsit sortint (ja ho fa per defecte)

### 2. Connectar i executar setup

```bash
ssh -i ~/.ssh/id_ed25519 opc@<IP-PUBLICA-VM>
# (o ubuntu@ si has triat Ubuntu)

# Descarregar i executar setup
git clone https://github.com/albertolive/nowcast-cardedeu.git
cd nowcast-cardedeu/oracle-cloud
chmod +x setup.sh run-predict.sh
./setup.sh
```

### 3. Configurar secrets

```bash
nano ~/nowcast-cardedeu/.env
# Emplenar:
#   TELEGRAM_BOT_TOKEN=xxx
#   TELEGRAM_CHAT_ID=xxx
#   METEOCAT_API_KEY=xxx
#   AEMET_API_KEY=xxx
```

### 4. Configurar git push

Triar **una** opció:

**Opció A — SSH key (recomanat):**
```bash
ssh-keygen -t ed25519 -C "nowcast-oracle-vm"
cat ~/.ssh/id_ed25519.pub
# Copiar i afegir a GitHub → Settings → SSH and GPG keys
cd ~/nowcast-cardedeu
git remote set-url origin git@github.com:albertolive/nowcast-cardedeu.git
```

**Opció B — Personal Access Token:**
```bash
cd ~/nowcast-cardedeu
git remote set-url origin https://<TOKEN>@github.com/albertolive/nowcast-cardedeu.git
```

### 5. Verificar

```bash
# Prova manual
sudo systemctl start nowcast-predict.service
journalctl -u nowcast-predict.service --no-pager -n 50

# Comprovar timer
systemctl list-timers nowcast-predict.timer

# Veure següent execució
systemctl status nowcast-predict.timer
```

## Què queda a GitHub Actions?

Després de la migració, GitHub Actions només executa:

| Job | Freqüència | Minuts/mes estimats |
|-----|-----------|---------------------|
| `daily_summary` | 1×/dia | ~30 min |
| `accuracy_report` | 1×/setmana | ~5 min |
| `retrain` | 1×/setmana (dg.) | ~20 min |
| `predict` (manual) | Només si la VM falla | 0 min (normalment) |
| **Total** | | **~55 min/mes** |

Vs. els ~1.700 min/mes anteriors (predict + retrain diari).

## Desactivar cron-job.org

Un cop la VM funcioni, desactiva el trigger de cron-job.org que invocava `workflow_dispatch` per la predicció.

## Monitorització

- **Logs**: `journalctl -u nowcast-predict.service -f`
- **Timer status**: `systemctl list-timers nowcast-*`
- **Fallades**: El script envia alertes Telegram si `predict_now.py` falla
- **Model updates**: La VM fa `git pull` a cada execució, agafant models retrenats automàticament
