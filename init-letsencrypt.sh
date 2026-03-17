#!/bin/bash

# Let's Encrypt Setup Script für chris-hesse.com

domain="chris-hesse.com"
email="your-email@example.com"  # WICHTIG: Hier deine E-Mail eingeben!
rsa_key_size=4096
certbot_staging=0  # 0 = Production, 1 = Staging (zum Testen)

echo "🔐 Starte Let's Encrypt Setup für $domain ..."

# Verzeichnisse anlegen
mkdir -p certbot/conf certbot/www

# Dummy Zertifikat erzeugen (damit Nginx starten kann)
if [ ! -f "./certbot/conf/live/$domain/fullchain.pem" ]; then
    echo "📝 Erstelle dummy Zertifikat..."
    docker compose run --rm --entrypoint "\
        openssl req -x509 -nodes -newkey rsa:1024 -days 1 \
        -keyout /etc/letsencrypt/live/$domain/privkey.pem \
        -out /etc/letsencrypt/live/$domain/fullchain.pem \
        -subj '/CN=$domain'" certbot
fi

# Nginx starten
echo "🚀 Starte Nginx..."
docker compose up -d nginx

# Echtes Zertifikat erstellen
echo "🔓 Beantrage Let's Encrypt Zertifikat..."
docker compose run --rm --entrypoint "\
    certbot certonly --webroot -w /var/www/certbot \
    --email $email \
    --agree-tos \
    --no-eff-email \
    -d $domain -d www.$domain" certbot

if [ $? -eq 0 ]; then
    echo "✅ Zertifikat erfolgreich erstellt!"
    echo "🔄 Lade Nginx Konfiguration neu..."
    docker compose exec nginx nginx -s reload
    echo "✅ Alles fertig! HTTPS ist aktiv auf https://$domain"
else
    echo "❌ Fehler bei der Zertifikatserstellung. Checke die DNS-Records!"
fi
