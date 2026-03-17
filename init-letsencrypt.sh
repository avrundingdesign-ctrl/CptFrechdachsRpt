#!/bin/bash

domain="chris-hesse.com"
email="your-email@example.com"  # WICHTIG: Hier deine E-Mail eingeben!

echo "🔐 Starte Let's Encrypt Setup für $domain ..."

# Verzeichnisse anlegen
mkdir -p certbot/conf/live/$domain certbot/www

# Dummy Zertifikat erzeugen (damit Nginx starten kann)
if [ ! -f "./certbot/conf/live/$domain/fullchain.pem" ]; then
    echo "📝 Erstelle dummy Zertifikat..."
    docker run --rm \
        -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
        certbot/certbot \
        certonly --standalone \
        --register-unsafely-without-email \
        --agree-tos \
        -d $domain \
        --staging
fi

# Nginx starten
echo "🚀 Starte Nginx..."
docker-compose up -d nginx

sleep 3

# Echtes Zertifikat erstellen
echo "🔓 Beantrage Let's Encrypt Zertifikat..."
docker run --rm \
    -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
    -v "$(pwd)/certbot/www:/var/www/certbot" \
    certbot/certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email $email \
    --agree-tos \
    --no-eff-email \
    --force-renewal \
    -d $domain

if [ $? -eq 0 ]; then
    echo "✅ Zertifikat erfolgreich erstellt!"
    docker-compose exec nginx nginx -s reload
    echo "✅ Alles fertig! HTTPS ist aktiv auf https://$domain"
else
    echo "❌ Fehler! DNS korrekt gesetzt? Test: nslookup $domain"
fi
