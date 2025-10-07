source venv/bin/activate

docker container ls
OPENRAG_ADDR=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' openrag-openrag-cpu-1`
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' openrag-openrag-cpu-1

docker logs openrag-openrag-cpu-1

echo "before"${OPENRAG_ADDR}"after"

#curl http://${OPENRAG_ADDR}:8080/health_check && echo Health is ok
#echo $?
while ! curl -fs "${OPENRAG_ADDR}:${PORT}/health_check" ;
do
  echo "Waiting for OpenRag to start at ${OPENRAG_ADDR}:${PORT}"
  sleep 10s
  OPENRAG_ADDR=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' openrag-openrag-cpu-1`
  docker container ls
done


python3 utility/data_indexer.py \
    -u http://${OPENRAG_ADDR}:8080 \
    -d .github/workflows/data/simplewiki-500/ \
    -p simplewiki-500

.github/workflows/mini/wait_for_tasks_completed.sh openrag-openrag-cpu-1 8080 500

