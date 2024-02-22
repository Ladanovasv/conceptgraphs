docker build . \
             -t concept_graphs \
             --build-arg UID=${UID} \
             --build-arg GID=${UID}