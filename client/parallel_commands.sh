#!/bin/bash

for num in $(seq $1 $2); do {
  echo "Process \"$num\" started";
  python3 client.py $num & pid=$!
  PID_LIST+=" $pid";
} done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";
