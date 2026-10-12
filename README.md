# Survey Dev Analytic

Dashboard analisis subset Stack Overflow Developer Survey. Aplikasi membaca `so_dev_survey.csv`, menyimpannya ke SQLite, lalu menampilkan ringkasan, kompensasi, teknologi, dan adopsi AI.

## Requirement

- Windows, macOS, atau Linux
- Python 3.11 atau lebih baru
- Koneksi internet saat instalasi paket

Paket Python yang dipakai:

| Paket | Versi minimum |
| --- | --- |
| pandas | 2.2 |
| plotly | 5.24 |
| streamlit | 1.40 |
| matplotlib | 3.8 |

Daftar lengkap ada di `dashboard/requirements.txt`.

Berkas data `so_dev_survey.csv` harus berada di folder yang sama dengan folder `dashboard`. Aplikasi membuat `dashboard/survey.db` sendiri saat pertama kali dijalankan, dan membangun ulang database jika CSV lebih baru.

## Instalasi

Dari folder proyek ini:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dashboard\requirements.txt
```

Di macOS atau Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r dashboard/requirements.txt
```

## Menjalankan

Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

macOS atau Linux:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Buka http://localhost:8501. Filter di samping yang dikosongkan berarti semua nilai pada kolom itu ikut dihitung.

Di tab **Ringkasan** ada tombol unduh PNG. Tanpa filter, gambar memakai judul, label, satuan, dan warna yang sama dengan laporan. Filter mengubah angka di dalam gambar.
