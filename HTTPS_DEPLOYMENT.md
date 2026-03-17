# 🔐 HTTPS Deployment Guide für chris-hesse.com

## Step 1: Domain DNS einrichten (Squarespace)

1. Login bei Squarespace
2. Gehe zu "Settings" → "Domains" → "chris-hesse.com"
3. Klick auf "Edit DNS Records" oder "Advanced"
4. Füge einen **A-Record** hinzu:
   - **Name:** @ (oder leer)
   - **Type:** A
   - **Value:** `deine-hetzner-ip` (z.B. `123.45.67.89`)
   - Speichern

5. Warte **5-15 Minuten**, damit die DNS-Änderung propagiert

Teste das:
```bash
nslookup chris-hesse.com
# Sollte deine IP anzeigen
```

---

## Step 2: Auf dem Server vorbereiten

```bash
cd ~/dartvision

# Script ausführbar machen
chmod +x init-letsencrypt.sh

# WICHTIG: E-Mail in init-letsencrypt.sh ändern!
# nano init-letsencrypt.sh
# Zeile: email="deine-email@example.com"
```

---

## Step 3: Let's Encrypt Setup ausführen

```bash
./init-letsencrypt.sh
```

Das Script macht:
1. Verzeichnisse anlegen
2. Dummy-Zertifikat erzeugen
3. Nginx starten
4. Echtes Let's Encrypt Zertifikat beantragen
5. Nginx neuladen

**Wenn alles gut geht:**
```
✅ Zertifikat erfolgreich erstellt!
✅ Alles fertig! HTTPS ist aktiv auf https://chris-hesse.com
```

---

## Step 4: Alle Services starten

```bash
docker compose up -d
```

Jetzt läuft:
- ✅ **Dartvision API** auf Port 5000 (intern)
- ✅ **Nginx** auf Port 80 (HTTP → HTTPS Redirect)
- ✅ **Nginx** auf Port 443 (HTTPS)
- ✅ **Certbot** (Auto-Renewal des Zertifikats)

---

## Step 5: Testen

```bash
# HTTP sollte zu HTTPS weiterleiten
curl -i http://chris-hesse.com/
# → HTTP 301 Redirect zu https://chris-hesse.com/

# HTTPS sollte funktionieren
curl https://chris-hesse.com/
# → {"status": "online", "service": "DartVision API", ...}
```

---

## Deine App updaten

In deiner App (iOS/Android) ändere den Endpoint zu:

```
https://chris-hesse.com/upload
```

Nicht mehr `http://ip:5000`!

---

## Logs checken

```bash
# Nginx Logs
docker compose logs nginx

# Certbot Logs (Auto-Renewal)
docker compose logs certbot

# Dartvision API
docker compose logs dartvision
```

---

## Automatisches Zertifikat-Renewal

Certbot prüft täglich, ob das Zertifikat erneuert werden muss. Falls es in 30 Tagen abläuft, wird es automatisch erneuert. ✅

---

## Troubleshooting

**"Connection refused" oder "DNS nicht gefunden?"**
- Warte 10-15 Minuten nach DNS-Änderung
- Checke: `nslookup chris-hesse.com`
- Starte Nginx neu: `docker compose restart nginx`

**"Invalid certificate"**
- Checke ob die Domain funktioniert: `ping chris-hesse.com`
- Logs: `docker compose logs certbot`

**Port 80 oder 443 belegt?**
- Linux: `sudo lsof -i :80` oder `:443`
- Kill: `sudo kill -9 <PID>`

---

## Sicherheit Checklist

- ✅ HTTPS aktiviert (SSL/TLS)
- ✅ Auto-Redirect HTTP → HTTPS
- ✅ Zertifikat-Auto-Renewal
- ✅ CORS aktiviert für App-Zugriff
- ✅ `client_max_body_size 100M` (für große Bilder)

Deine öffentliche App ist jetzt sicher! 🎉
