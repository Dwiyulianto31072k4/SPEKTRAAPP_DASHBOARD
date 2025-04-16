#!/bin/bash

echo "🔄 Cek status..."
git status

echo "➕ Menambahkan semua file..."
git add .

echo "📝 Masukkan pesan commit (kosong = batal):"
read msg

if [[ -z "$msg" ]]; then
  echo "❌ Commit dibatalkan karena tidak ada pesan."
  exit 1
fi

git commit -m "$msg"

echo "🚀 Push ke GitHub..."
git push

echo "✅ Selesai dipush ke GitHub!"
