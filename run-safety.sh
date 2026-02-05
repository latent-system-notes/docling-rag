sudo docker run --gpus all \
-it \
--name docling \
-e PATH="/opt/venv/bin:$PATH" \
-e VIRTUAL_ENV="/opt/venv" \
-v "/home/user/workspace/data:/workspace/data" \
-v "/home/user/workspace/docs:/workspace/docs" \
-v "/home/user/workspace/docling-rag:/workspace/app" \
-p "9090:9090" \
nvcr.io/nvidia/docling:0.0.1-rc3