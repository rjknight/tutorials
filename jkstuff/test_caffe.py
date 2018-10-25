#!/usr/bin/python

from caffe2.python import workspace

import caffe2.python.core

import threading
import caffe2_multithreading

NUM_THREADS = 1

def worker():
    caffe2_multithreading.main()

def run_workers():
    threads = [threading.Thread(name="Thread{}".format(i), target=worker) for i in range(NUM_THREADS)]
    for thread in threads:
        thread.start()

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
run_workers()

