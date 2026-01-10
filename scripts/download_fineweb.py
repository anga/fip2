#!/usr/bin/env python3
"""
Descarga FineWeb-Edu sample-10BT en múltiples archivos pequeños.

Uso:
    python scripts/download_fineweb.py                        # 1M docs en archivos de 50MB
    python scripts/download_fineweb.py --limit 100000         # 100K docs
    python scripts/download_fineweb.py --file-size 100        # Archivos de 100MB
    python scripts/download_fineweb.py --output-dir data/     # Directorio custom
"""
import argparse
import gc
import os
import sys
from itertools import islice

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' no está instalado.")
    print("Instala con: pip install datasets")
    sys.exit(1)


def batch_iterator(iterator, batch_size):
    """Genera batches de un iterador sin cargar todo en memoria."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def main():
    parser = argparse.ArgumentParser(description="Descarga FineWeb-Edu para FIP2")
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Directorio de salida (default: data/)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=1_000_000,
        help="Número máximo de documentos a descargar (default: 1M)"
    )
    parser.add_argument(
        "--file-size", "-s",
        type=int,
        default=50,
        help="Tamaño máximo por archivo en MB (default: 50MB)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1_000,
        help="Docs por batch de procesamiento (default: 1K)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Descargar dataset completo (~10B tokens)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Longitud mínima de texto en caracteres (default: 100)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FineWeb-Edu Downloader para FIP2")
    print("=" * 60)

    limit = None if args.full else args.limit
    max_file_bytes = args.file_size * 1024 * 1024

    if limit:
        print(f"Límite: {limit:,} documentos")
    else:
        print("Descargando dataset completo")

    print(f"Directorio: {args.output_dir}/")
    print(f"Tamaño max por archivo: {args.file_size}MB")
    print("-" * 60)

    # Crear directorio
    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar dataset en streaming
    print("\nConectando a HuggingFace...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    # Contadores globales
    total_docs = 0
    total_chars = 0
    total_skipped = 0
    file_num = 0

    # Estado del archivo actual
    current_file = None
    current_file_bytes = 0

    def open_new_file():
        nonlocal file_num, current_file, current_file_bytes
        if current_file:
            current_file.close()
        file_num += 1
        path = os.path.join(args.output_dir, f"fineweb_{file_num:04d}.txt")
        current_file = open(path, "w", encoding="utf-8")
        current_file_bytes = 0
        print(f"\nCreando: {path}")
        return current_file

    # Abrir primer archivo
    open_new_file()

    # Iterar dataset
    data_iter = iter(dataset)

    for batch in batch_iterator(data_iter, args.batch_size):
        for example in batch:
            text = example.get("text", "")

            # Filtrar textos muy cortos
            if len(text) < args.min_length:
                total_skipped += 1
                continue

            # Preparar texto
            text_clean = text.strip() + "\n\n"
            text_bytes = len(text_clean.encode("utf-8"))

            # ¿Necesitamos nuevo archivo?
            if current_file_bytes + text_bytes > max_file_bytes:
                open_new_file()

            # Escribir
            current_file.write(text_clean)
            current_file_bytes += text_bytes
            total_docs += 1
            total_chars += len(text)

            # Verificar límite
            if limit and total_docs >= limit:
                break

        # Flush periódico
        current_file.flush()

        # Mostrar progreso
        total_mb = total_chars / (1024 * 1024)
        print(f"  Docs: {total_docs:,} | Archivos: {file_num} | Total: {total_mb:.1f}MB", end="\r")

        # Liberar memoria
        del batch
        gc.collect()

        # Verificar límite global
        if limit and total_docs >= limit:
            break

    # Cerrar último archivo
    if current_file:
        current_file.close()

    # Estadísticas finales
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith(".txt")
    )
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)

    print("\n\n" + "=" * 60)
    print("Descarga completada!")
    print("=" * 60)
    print(f"  Documentos: {total_docs:,}")
    print(f"  Filtrados:  {total_skipped:,}")
    print(f"  Archivos:   {file_num}")
    if total_size_gb >= 1:
        print(f"  Tamaño:     {total_size_gb:.2f} GB")
    else:
        print(f"  Tamaño:     {total_size_mb:.1f} MB")
    print(f"  Directorio: {args.output_dir}/")
    print("\nPara entrenar:")
    print(f"  python -m fip2.main train --data {args.output_dir}")


if __name__ == "__main__":
    main()
