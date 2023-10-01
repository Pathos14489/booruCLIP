# BooruCLIP

Very early prototype, but works pretty good imo.

Make sure you run the CLIP-as-service server with a model set for 768x768 images. If you want to change what size the images are, set booruCLIP.size to the image size you want. Default is 768x768. I've included the tensor-flow.yaml that I believe I used when testing as an example.

Run set your API key for Derpibooru in train.py and it'll start scraping and training the embedding model when you run it. You should be safe to pause and resume the training as needed.

After that, run gr_demo.py to test the model.
