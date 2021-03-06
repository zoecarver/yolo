## The notebook

You probably want to look [here](https://github.com/zoecarver/yolo/blob/master/main.ipynb).

## Example

![example](/Users/zoe/Developer/yolo/yolo/doc/example.png)

## Things you need

Download the weights:

```bash
wget http://pjreddie.com/media/files/yolo.weights
```

## Questions

1. Why does this particular set of layers work so well together. 
2. How does the loss function work

*most questions are at the top of sections in the notebook*

## Credits

This project was heavily inspired by [this repo](https://github.com/rodrigo2019/keras-yolo2/tree/trained-weights) and [this article](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/) and obviosly is based on [the yolo network](https://pjreddie.com/darknet/yolov2/). 

## Random

**miliseconds per evaluation:** ~350

you can see the model images in the `doc` directory. 

## Roadmap

- [x] Implement layers of network (Q1)
- [x] Pipeline for image processing
- [X] Fix weight reader
  - Once we are here we *should* be able to just load any config file and have it work
- [ ] Make config reader "standard"
- [ ] Understand loss function (Q2)
- [ ] Implementing loss function and training
- [ ] Improve upon network
- [ ] Documentation of network
