

```
    docker build --platform=linux/amd64 -t emg_ann_service .
```

```
    docker run --platform=linux/amd64 -p 8000:8000 emg_ann_service
```