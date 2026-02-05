sudo docker run --gpus all \
-it \
-w "/workspace/app" \
-e PATH="/opt/venv/bin:$PATH" \
-e VIRTUAL_ENV="/opt/venv" \
-v "/home/user/workspace/data/techpub:/workspace/data" \
-v "/mnt/techpub-docs/3. General Military Publications:/workspace/docs" \
-v "/home/user/workspace/docling-rag:/workspace/app" \
-p "9090:9090" \
--rm \
nvcr.io/nvidia/docling:0.0.1-rc3


##### ==== DAEMON RUN

sudo docker run \
--gpus all \
-d \
-w "/workspace/app" \
-e PATH="/opt/venv/bin:$PATH" \
-e VIRTUAL_ENV="/opt/venv" \
-v "/home/user/workspace/data/techpub:/workspace/data" \
-v "/mnt/techpub-docs/3. General Military Publications:/workspace/docs" \
-v "/home/user/workspace/docling-rag:/workspace/app" \
-p "9090:9090" \
--rm \
--name docling-techpub-ingest \
nvcr.io/nvidia/docling:0.0.1-rc3 \
python -m cli.cli ingest techpub