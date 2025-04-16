#!/bin/bash

echo "🔄 Cek status..."
git status

echo "➕ Menambahkan semua file..."
git add .

echo "📝 Masukkan pesan commit:"
read msg

git commit -m "$msg"

echo "🚀 Push ke GitHub..."
git push -u origin main

echo "✅ Selesai dipush ke GitHub!"
