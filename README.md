# Federated Learning with Flower framework

## How to run

**Note: This implementation runs on Mac. Then the device is set "mps". You should change the device according to your hardware.

### Create dataset

This step will create iid or non-iid datasets for clients from MNIST dataset

For example: To reproduce the experiment desribed in the (paper)[].
```bash
python create_dataset.py --type noniid\
                         --num_clients 100\
                         --num_shards 200
```

### Approach 1: Start server-clients manually

Start the server first

```bash
python server.py --rnd 30
```

Then open the other terminals to start each client.
For example, start the client 1.

```bash
python client.py --cid 1\ 
                --data_dir datasets/fl_mnist_noniid
```

### Approach 2: Start the simulation with many clients

```bash
python simulation.py --rnd 50
```

## Some resources to start Federated Learning

- Introduction Slides for FL: https://namhoonlee.github.io/courses/optml/s14-fl.pdf
- Video tutorial: https://www.youtube.com/watch?v=nBGQQHPkyNY
- Flower tutorial: https://www.youtube.com/watch?v=Ky6TicaPfVI&t=2656s

## Todo

[ ] Update some results on MNIST