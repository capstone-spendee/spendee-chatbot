# Gunakan image Python sebagai base
FROM python:3.10-slim

# Atur working directory di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port default Flask
EXPOSE 8080

# Jalankan aplikasi
CMD ["gunicorn", "-b", ":8080", "app:app"]