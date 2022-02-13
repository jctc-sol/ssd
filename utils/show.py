def drawSample(sample):
    """
    Visualize the given sample by showing the image and
    draws bounding boxes of target objects in the image
    """
    img  = sample['image']
    if isinstance(img, torch.Tensor):
        tf = transforms.ToPILImage()
        img = tf(img)
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(sample['boxes'].shape[0]):
        # go through all boxes & draw
        # bbox coordinates are provides in terms of [x,y,width,height], and
        # box coordinates are measured from the top left image corner and are 0-indexed
        box = list(sample['boxes'][i,:])
        x0, y0 = box[0], box[1]
        x1, y1 = x0+box[2], y0+box[3]
        draw.rectangle([x0,y0,x1,y1], outline='orange', width=4)
#         # TODO: draw instance segmentation maps
#         # go through all segmaps & draw
#         segmap = sample['segs'][i][0]
#         draw.polygon(segmap, fill=(0, 255, 255, 125))
    plt.imshow(img); plt.axis('off')

    
def show_augmented_samples(ds, sample_idx=0, n=10):
    """
    Helper function to visualize the same image sample
    specified by `sample_idx`, `n` number of times; 
    each subjected to data augmentations applied to 
    dataset `ds`.
    """
    samples_per_row = 5
    num_rows = int(n/samples_per_row) if n%samples_per_row==0 else int(n/samples_per_row)+1
    plt.figure(figsize=(16, 2*num_rows))
    for i in range(n):
        plt.subplot(num_rows, samples_per_row, i+1)
        drawSample(ds[sample_idx])    
