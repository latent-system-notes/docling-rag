==============================================================================
  DOCLING-RAG  --  Air-Gap Deployment Guide
==============================================================================

OVERVIEW
--------
Two machines on the same LAN:
  - SparkDGX (Linux aarch64, NVIDIA GPU) --> runs docling-rag app containers
  - Windows Server 2022 (amd64)          --> runs PostgreSQL + pgvector

Transfer bundle contents:
  docling-rag-arm64.tar      App image for SparkDGX (~15-20 GB)
  pgvector-pg17-amd64.tar    PostgreSQL image for Windows Server (~500 MB)
  docling-rag/               Application code (this repo)
    docker/init.sql           Database schema
    .env.techpub              Config for techpub service
    .env.safety               Config for safety service

==============================================================================
A. WINDOWS SERVER 2022 -- PostgreSQL Setup
==============================================================================

1. Load the image:

   docker load -i pgvector-pg17-amd64.tar

2. Copy init.sql to a local directory:

   mkdir C:\docling-postgres
   copy docling-rag\docker\init.sql C:\docling-postgres\init.sql

3. Run PostgreSQL:

   docker run -d --name docling-postgres ^
     -e POSTGRES_DB=docling_rag ^
     -e POSTGRES_USER=docling ^
     -e POSTGRES_PASSWORD=docling ^
     -v pgdata:/var/lib/postgresql/data ^
     -v C:\docling-postgres\init.sql:/docker-entrypoint-initdb.d/init.sql:ro ^
     -p 5432:5432 ^
     --restart unless-stopped ^
     pgvector/pgvector:pg17

4. Verify:

   docker exec docling-postgres psql -U docling -d docling_rag -c "\dt"
   --> Should show: chunks table

   docker exec docling-postgres psql -U docling -d docling_rag -c "\dx"
   --> Should show: vector extension

5. Open firewall port 5432:

   New-NetFirewallRule -DisplayName "PostgreSQL" -Direction Inbound -Protocol TCP -LocalPort 5432 -Action Allow

6. Note the server IP:

   ipconfig
   --> e.g. 192.168.1.100

==============================================================================
B. SPARKDGX -- Application Setup
==============================================================================

1. Load the app image:

   docker load -i docling-rag-arm64.tar

2. Deploy app code:

   cp -r docling-rag /home/user/workspace/docling-rag

3. Configure .env files:

   Edit .env.techpub and .env.safety -- set POSTGRES_HOST to the Windows
   Server LAN IP:

     POSTGRES_HOST=192.168.1.100    # <-- Windows Server LAN IP
     POSTGRES_PORT=5432

4. Test connectivity:

   docker run --rm \
     -v /home/user/workspace/docling-rag:/workspace/app \
     --env-file /home/user/workspace/docling-rag/.env.techpub \
     docling-rag:latest \
     python -m cli stats techpub

5. Ingest documents:

   docker run --rm \
     -v /home/user/workspace/docling-rag:/workspace/app \
     -v /path/to/techpub-docs:/workspace/docs \
     --env-file /home/user/workspace/docling-rag/.env.techpub \
     --gpus all \
     docling-rag:latest \
     python -m cli ingest techpub

6. Run MCP server -- techpub (long-running):

   docker run -d --name docling-techpub \
     -v /home/user/workspace/docling-rag:/workspace/app \
     -v /path/to/techpub-data:/workspace/data \
     -v "/path/to/techpub-docs:/workspace/docs" \
     --env-file /home/user/workspace/docling-rag/.env.techpub \
     --gpus all \
     -p 9090:9090 \
     --restart unless-stopped \
     docling-rag:latest \
     python -m cli mcp techpub

7. Run MCP server -- safety (long-running):

   docker run -d --name docling-safety \
     -v /home/user/workspace/docling-rag:/workspace/app \
     -v /path/to/safety-data:/workspace/data \
     -v /path/to/safety-docs:/workspace/docs \
     --env-file /home/user/workspace/docling-rag/.env.safety \
     --gpus all \
     -p 9091:9090 \
     --restart unless-stopped \
     docling-rag:latest \
     python -m cli mcp safety

==============================================================================
C. VERIFICATION
==============================================================================

From SparkDGX:

   docker exec docling-techpub python -m cli stats techpub
   docker exec docling-techpub python -m cli list techpub
   docker exec docling-techpub python -m cli query techpub "test query"

==============================================================================
D. UPDATING APP CODE (post-deployment)
==============================================================================

Since code is volume-mounted, updating is simple:

   1. SCP new code to /home/user/workspace/docling-rag/ on SparkDGX
   2. Restart the container:
      docker restart docling-techpub
      docker restart docling-safety

No image rebuild needed for code-only changes.
Only rebuild the image if requirements.txt or models change.

==============================================================================
E. TROUBLESHOOTING
==============================================================================

Problem: "connection refused" to Windows Server
Fix:     Check firewall rule, PostgreSQL container running, correct IP

Problem: 'role "docling" does not exist'
Fix:     Delete pgdata volume and recreate:
         docker volume rm pgdata
         Then re-run the docker run command from step A.3

Problem: Container starts but models not found
Fix:     Models are in /opt/models inside the image.
         Do NOT mount a volume over /opt/models.

Problem: Code changes not reflected
Fix:     Restart the container after updating mounted code:
         docker restart docling-techpub

Problem: PostgreSQL init.sql not applied
Fix:     init.sql only runs on first start with empty pgdata volume.
         To re-init: stop container, docker volume rm pgdata, restart.
