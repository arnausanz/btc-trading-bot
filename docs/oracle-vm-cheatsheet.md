# Oracle Cloud VM — Guia de Comandes

> **Valors que necessites tenir a mà:**
> - **CLAU PRIVADA SSH:** `~/.ssh/id_ed25519`
> - **IP PÚBLICA DE LA VM:** `79.76.110.205`
> - **USUARI:** `ubuntu`

---

## 1. Connectar-se a la VM

Obre un terminal al Mac i executa:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@79.76.110.205
```

Per sortir de la VM:

```bash
exit
```

> ⚠️ Cada secció indica explícitament on s'ha d'executar: **al Mac** o **dins de la VM** (connectat per SSH).

---

## 2. Copiar fitxers del Mac a la VM

> ⚠️ Aquestes comandes s'executen des d'un terminal del **Mac**, NO des de la VM.

### Copiar els models entrenats
```bash
scp -i ~/.ssh/id_ed25519 -r /Users/arnau/Documents/Projectes/btc-trading-bot/models ubuntu@79.76.110.205:~/trading-bot/models
```

### Copiar el fitxer .env (si el modifiques en local)
```bash
scp -i ~/.ssh/id_ed25519 /Users/arnau/Documents/Projectes/btc-trading-bot/.env ubuntu@79.76.110.205:~/trading-bot/.env
```

### Copiar el docker-compose.yml (si el modifiques en local)
```bash
scp -i ~/.ssh/id_ed25519 /Users/arnau/Documents/Projectes/btc-trading-bot/docker-compose.yml ubuntu@79.76.110.205:~/trading-bot/docker-compose.yml
```

---

## 3. Actualitzar el codi des de GitHub

> ⚠️ Aquestes comandes s'executen **dins de la VM**.

### Pull de la branca main (actualització normal)
```bash
cd ~/trading-bot
git pull origin main
```

### Comprovar en quina branca estàs i l'estat del repo
```bash
cd ~/trading-bot
git status
git log --oneline -5
```

---

## 4. Gestió de Docker

> ⚠️ Aquestes comandes s'executen **dins de la VM**.

### Arrencar tots els serveis
```bash
cd ~/trading-bot
docker compose up -d
```

### Aturar tots els serveis
```bash
cd ~/trading-bot
docker compose down
```

### Veure l'estat dels serveis (healthy / running / etc.)
```bash
cd ~/trading-bot
docker compose ps
```

### Veure els logs en temps real
```bash
cd ~/trading-bot
docker compose logs -f
```

### Veure els logs d'un servei concret (ex: la BD)
```bash
cd ~/trading-bot
docker compose logs -f db
```

### Reiniciar un servei concret
```bash
cd ~/trading-bot
docker compose restart db
```

### Actualitzar i reiniciar després d'un git pull
```bash
cd ~/trading-bot
git pull origin main
docker compose down
docker compose up -d
```

---

## 5. Comprovar l'estat de la VM

> ⚠️ Aquestes comandes s'executen **dins de la VM**.

### RAM disponible
```bash
free -h
```

### Espai en disc
```bash
df -h /
```

### Espai ocupat per carpetes del projecte
```bash
du -sh ~/trading-bot/*
```

### Espai ocupat pels models
```bash
du -sh ~/trading-bot/models
```

### Processos de Docker i consum de recursos
```bash
docker stats
```
> Prem `Ctrl+C` per sortir.

---

## 6. Connexió a la base de dades (TimescaleDB)

Necessites **dues terminals** obertes al Mac simultàniament.

### Terminal 1 — Obre el túnel SSH (deixa-la oberta)
```bash
ssh -i ~/.ssh/id_ed25519 -L 5432:localhost:5432 ubuntu@79.76.110.205 -N
```
> Aquesta terminal quedarà bloquejada sense mostrar res. És normal, el túnel està actiu.

### Terminal 2 — Connecta't a la BD
```bash
psql -h localhost -U btc_user -d btc_trading
```

### Comandes útils dins de psql
```sql
-- Llistar totes les taules
\dt

-- Veure la versió de PostgreSQL
SELECT version();

-- Sortir de psql
\q
```

> ⚠️ Quan acabis, tanca el túnel a la Terminal 1 prement `Ctrl+C`.

---

## 7. Flux de treball habitual (deploy d'una nova versió)

Quan hagis fet canvis en local i els vulguis desplegar a la VM:

**Al Mac (local):**
```bash
# 1. Estàs a develop, fas merge a main
git checkout main
git merge develop
git push origin main
```

**A la VM (connectat per SSH):**
```bash
# 2. Atura la demo abans de res
sudo systemctl stop trading-demo

# 3. Actualitza el codi
cd ~/trading-bot
git pull origin main

# 4. Si has canviat el docker-compose.yml, reinicia els serveis Docker
docker compose down
docker compose up -d

# 5. Si hi ha noves migracions de BD
poetry run alembic upgrade head

# 6. Comprova que Docker funciona
docker compose ps

# 7. Torna a arrencar la demo
sudo systemctl start trading-demo
sudo systemctl status trading-demo
```

**Si has entrenat nous models al Mac:**
```bash
# Des del Mac (terminal separada)
scp -i ~/.ssh/id_ed25519 -r /Users/arnau/Documents/Projectes/btc-trading-bot/models ubuntu@79.76.110.205:~/trading-bot/models
```

---

## 8. Gestió de la Demo (systemd)

> ℹ️ La demo corre com un **servei del sistema** (systemd). Això significa que s'arrenca automàticament quan la VM arrenca, es reinicia sola si cau, i segueix corrent quan surts de la terminal. **No cal fer res especial per mantenir-la viva.**

### Comprovar l'estat de la demo
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
sudo systemctl status trading-demo
```

Ha de mostrar **active (running)** en verd.

### Veure els logs de la demo en temps real
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
journalctl -fu trading-demo
```

> Prem `Ctrl+C` per sortir dels logs. La demo **no s'atura**, només deixes de veure els logs.

### Aturar la demo manualment
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
sudo systemctl stop trading-demo
```

### Arrencar la demo manualment
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
sudo systemctl start trading-demo
```

### Reiniciar la demo (després d'un deploy)
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
sudo systemctl restart trading-demo
```

### Desactivar la demo permanentment (que no arranqui amb la VM)
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
sudo systemctl disable trading-demo
sudo systemctl enable trading-demo  # per reactivar-la
```

---

## 9. Backup automàtic de la base de dades

> ℹ️ La VM fa backups automàtics cada dia a les **3:00 AM** (hora de mínim tràfic de mercat). Cada backup és un fitxer comprimit `.sql.gz` que conté **totes les dades** de totes les taules. Els backups dels últims **7 dies** es conserven automàticament — els més antics s'esborren sols a les 3:15 AM del dia següent. No cal fer res manualment.

### Comprovar que el cron està actiu
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
# Comprova que el servei cron està corrent
sudo systemctl status cron

# Comprova que les línies del crontab estan configurades
crontab -l
```

El `systemctl status cron` ha de mostrar **active (running)**. El `crontab -l` ha de mostrar les dues línies del backup.

### Veure els backups disponibles a la VM
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
ls -lh ~/backup_*.sql.gz
```

Veuràs els fitxers disponibles amb el seu pes. Exemple:
```
-rw-r--r-- backup_20260320_0300.sql.gz   45M
-rw-r--r-- backup_20260321_0300.sql.gz   46M
```

### Descarregar tots els backups al Mac
> 📍 Executa des d'un terminal del **Mac** (NO des de la VM)

```bash
scp -i ~/.ssh/id_ed25519 ubuntu@79.76.110.205:~/backup_*.sql.gz ~/Downloads/
```

### Descarregar un backup concret al Mac
> 📍 Executa des d'un terminal del **Mac** (NO des de la VM)

```bash
# Substitueix el nom pel fitxer que vulguis
scp -i ~/.ssh/id_ed25519 ubuntu@79.76.110.205:~/backup_20260321_0300.sql.gz ~/Downloads/
```

### Fer un backup manual puntual (opcional)
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
docker exec btc_trading_db pg_dump -U btc_user btc_trading | gzip > ~/backup_manual_$(date +%Y%m%d_%H%M).sql.gz
```

### Restaurar la BD des d'un backup (si cal)
> 📍 Executa **dins de la VM** (qualsevol directori)

```bash
# Substitueix el nom del fitxer pel que vulguis restaurar
gunzip -c ~/backup_20260321_0300.sql.gz | docker exec -i btc_trading_db psql -U btc_user btc_trading
```

> ⚠️ Això sobreescriu les dades actuals de la BD. Només fer-ho si cal recuperar d'un error.

---

## 10. Resolució de problemes habituals

### La demo no arrenca o falla
```bash
# Mira els logs detallats
journalctl -u trading-demo -n 50
```

### Docker no funciona i la demo falla
```bash
cd ~/trading-bot
docker compose ps
docker compose up -d
sudo systemctl restart trading-demo
```

### El servei no arrenca (status: exited)
```bash
cd ~/trading-bot
docker compose logs db   # substitueix 'db' pel servei que falla
```

### El disc s'omple
```bash
# Elimina imatges Docker no usades
docker system prune -a
```

### Perdut dins de la VM, tornar a home
```bash
cd ~
```

### Comprovar si la VM és accessible abans de connectar-se
```bash
# Des del Mac
ping 79.76.110.205
```
