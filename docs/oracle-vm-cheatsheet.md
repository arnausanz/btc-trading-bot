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

> ⚠️ Totes les comandes de les seccions 3, 4 i 5 s'executen **dins de la VM** (un cop connectat per SSH). Les de les seccions 2 i 6 s'executen **al Mac**.

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
# 2. Actualitza el codi
cd ~/trading-bot
git pull origin main

# 3. Si has canviat el docker-compose.yml, reinicia els serveis
docker compose down
docker compose up -d

# 4. Comprova que tot funciona
docker compose ps
```

**Si has entrenat nous models al Mac:**
```bash
# Des del Mac (terminal separada)
scp -i ~/.ssh/id_ed25519 -r /Users/arnau/Documents/Projectes/btc-trading-bot/models ubuntu@79.76.110.205:~/trading-bot/models
```

---

## 8. Resolució de problemes habituals

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
