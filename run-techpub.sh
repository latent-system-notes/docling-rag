##### ==== DAEMON RUN INGESTION

sudo docker run \
--gpus all \
--name techpub-ingestion \
-d \
-w "/workspace/app" \
-e PATH="/opt/venv/bin:$PATH" \
-e VIRTUAL_ENV="/opt/venv" \
-v "/home/user/workspace/data/techpub:/workspace/data" \
-v "/mnt/techpub-docs/6. TPO Shared:/workspace/docs" \
-v "/home/user/workspace/docling-rag:/workspace/app" \
--restart unless-stopped \
nvcr.io/nvidia/docling:0.0.1-rc3 \
python -m cli.cli ingest techpub

##### ==== DAEMON RUN MCP

sudo docker run \
--gpus all \
--name techpub-mcp \
-d \
-w "/workspace/app" \
-e PATH="/opt/venv/bin:$PATH" \
-e VIRTUAL_ENV="/opt/venv" \
-v "/home/user/workspace/data/techpub:/workspace/data" \
-v "/mnt/techpub-docs/5. Vendors Publications:/workspace/docs" \
-v "/home/user/workspace/docling-rag:/workspace/app" \
-p "9090:9090" \
--restart unless-stopped \
nvcr.io/nvidia/docling:0.0.1-rc3 \
python -m cli.cli mcp techpub