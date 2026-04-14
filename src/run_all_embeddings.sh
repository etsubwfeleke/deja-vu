#!/bin/bash
echo "=== Extracting all embeddings ==="

for model in clip resnet; do
    for split in banned reuploads negatives; do
        echo ""
        echo "--- $model / $split ---"
        python src/extract_embeddings.py \
            --input_dir data/$split \
            --output_path embeddings/${split}_${model}.npy \
            --model $model
    done
done

echo ""
echo "=== All done ==="
ls -lh embeddings/