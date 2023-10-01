import torch
import os
import json
import uuid
import time
import time
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from clip_client import Client
from docarray import Document, DocumentArray


class BooruCLIP:
    def __init__(self,base_directory,size = 768,clip = None):
        self.base_directory = base_directory
        self.embeddingCache = {}
        self.imagesEmbeddings = DocumentArray()
        self.embeddingsDir = base_directory + "embeddingCache/"
        self.convertedImagesDir = base_directory + "converted_images/"
        if size is None:
            size = 768 # Make sure this matches the recommended image size of the CLIP model you're using
        self.size = size

            
        if not os.path.exists(base_directory):
            os.mkdir(base_directory)
        if not os.path.exists(self.embeddingsDir):
            os.mkdir(self.embeddingsDir)
        if not os.path.exists(self.convertedImagesDir):
            os.mkdir(self.convertedImagesDir)
        print("Converted Images:",len(os.listdir(self.convertedImagesDir)))
        print("Embeddings:",len(os.listdir(self.embeddingsDir)))
        if clip is None:
            self.clip = Client('grpc://localhost:51000')
        else:
            self.clip = clip

        if os.path.exists(self.base_directory + "checkpoint.json"):
            self.load_checkpoint(self.base_directory + "checkpoint.json")
        else:
            print("Loading Embedding Cache")
            for file in tqdm(os.listdir(self.embeddingsDir)):
                if file.endswith(".json"):
                    with open(self.embeddingsDir + file) as f:
                        # print(f)
                        try:
                            data = json.load(f)
                            if type(data) == str:
                                data = json.loads(data)
                        except Exception as e:
                            print("Error loading embedding:"+file)
                            print(e)
                            pass
                        # print(data["tags"]["ID"])
                        data = json.dumps(data)
                        self.imagesEmbeddings.append(Document.from_json(data))
                        if type(data) is str:
                            data = json.loads(data)
                        self.embeddingCache[data["tags"]["ID"]] = data
                        # if blip2:
                        #     tags = data["tags"]["TAGS"]
                        #     tags = ", ".join(tags)
                        #     prompt = "Tags: " + tags + "\n"
                        #     prompt += "You are a visual captioning AI for blind people. You always caption every possible detail of an image, and never make up details about an image. Please caption the image using the image and the tags above as reference. Please make your caption as long as necessary to fully explain everything about the image, and make sure every tag is represented in your caption.\nDetailed Caption:"
                        #     image = Image.open(self.convertedImagesDir + data["tags"]["ID"] + ".jpg")
                        #     print("Prompting BLIP2 for image: " + data["tags"]["ID"])
                        #     output = self.blip2.answer(image,prompt)
                        #     print("BLIP2 Response: " + output)
                        #     llm_prompt = "Tags: clothes, green, eyes, hybrid, looking, at, you, oc, safe, simple, background, snake, wings, snake, pony, original, species, oc:razor, uniboop, artist:nhale\nImperfect Caption: a hybrid of a pony and a snake with green eyes looking at you\nYou are being used with a visual captioning accessory. Adjust yourself to be more relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. You have been provided the tags of an image, and a caption from a sometimes imperfect vision AI. Please correct the caption to be as accurate as possible. Please do not make up details about the image that are unlikely to be real based on the tags and te caption provided, however, if the provided caption is complete gibberish, please attempt to create a convincing caption for the image using only the tags. Use your best judgement.\nAdjusted Caption: A hybrid of a pegasus pony and a snake named Razor with green eyes looking at you drawn on a simple background by nhale\nTags: " + tags + "\nImperfect Caption:" + output + "\nYou are being used with a visual captioning accessory. Adjust yourself to be more relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. You have been provided the tags of an image, and a caption from a sometimes imperfect vision AI. Please correct the caption to be as accurate as possible. Please do not make up details about the image that are unlikely to be real based on the tags and te caption provided, however, if the provided caption is complete gibberish, please attempt to create a convincing caption for the image using only the tags. Use your best judgement.\nAdjusted Caption:"
                        #     llm_output = self.llama.generate_sync(llm_prompt, max_length=256, top_k=40, top_p=0.95, temperature=0.5)
                        #     print("LLAMA Response: " + llm_output)       
            self.save_checkpoint(self.base_directory + "checkpoint.json")
                        
        print("BooruCLIP Initialized")
    
    def save_checkpoint(self,path): # save to .json
        print("Saving Checkpoint")
        if os.path.exists(path):
            print("Checkpoint already exists, appending")
            pre_existing = []
            with open (path, "r") as f:
                for line in f:
                    pre_existing.append(json.loads(line)["tags"]["ID"])
            with open(path, 'w') as outfile:
                for key in tqdm(self.embeddingCache.keys()):
                    if key not in pre_existing:
                        outfile.write(json.dumps(self.embeddingCache[key]))
                        outfile.write("\n")
        else: 
            print("Checkpoint does not exist, creating")
            with open(path, 'w') as outfile:
                for key in tqdm(self.embeddingCache.keys()):
                    outfile.write(json.dumps(self.embeddingCache[key]))
                    outfile.write("\n")
                    
    def load_checkpoint(self,path): # load from .json
        print("Loading Checkpoint")
        pre_existing = []
        with open(path, "r") as json_file:
            for line in tqdm(json_file):
                data = json.loads(line)
                pre_existing.append(data["tags"]["ID"])
                if self.embeddingCache.get(data["tags"]["ID"]) is None:
                    self.embeddingCache[data["tags"]["ID"]] = data
                    self.imagesEmbeddings.append(Document.from_json(json.dumps(data)))
        embedding_files = os.listdir(self.embeddingsDir)
        embedding_files = [file.split(".")[0] for file in embedding_files]
        embedding_files = [id for id in embedding_files if id not in pre_existing]
        embedding_files = [id + ".json" for id in embedding_files]
        if len(embedding_files) is not len(pre_existing):
            for file in tqdm(embedding_files):
                file_id = file.split(".")[0]
                if file_id not in pre_existing:
                    # if file exists on disk but not in memory, load it
                    if os.path.exists(self.embeddingsDir + file):
                        with open(self.embeddingsDir + file) as f:
                            try:
                                data = json.load(f)
                                if type(data) == str:
                                    data = json.loads(data)
                            except Exception as e:
                                print("Error loading embedding:"+file)
                                print(e)
                                pass
                            self.embeddingCache[data["tags"]["ID"]] = data
                            self.imagesEmbeddings.append(Document.from_json(json.dumps(data)))
        
    def cachedEmbeddings(self):
        # return values from embeddingCache as a list
        return list(self.embeddingCache.values())
    def cachedEmbeddingsIDs(self):
        # return keys from embeddingCache as a list
        return list(self.embeddingCache.keys())
    def convertedImages(self):
        # return list of converted images
        return os.listdir(self.convertedImagesDir)
    def get_by_tags(self, tgs=[]):
        res = DocumentArray()
        startTime = time.time()
        print(tgs)
        if len(tgs) > 0:
            for doc in self.imagesEmbeddings:
                hasAll = True
                for tag in tgs:
                    if tag not in doc.tags["TAGS"]:
                        hasAll = False
                if hasAll:
                    res.append(doc)
        else:
            res = self.imagesEmbeddings
        endTime = time.time()
        print(str(len(res))+"| Tag Filtering Time: " + str(endTime - startTime))
        return res
    
    def query(self, text, top_k=100, top_p=0.5, tags=[]):
        sid = str(uuid.uuid4())
        startTime = time.time()
        text_embedding = self.clip.encode([text])[0]
        endTime = time.time()
        print("Text Encoding time: " + str(endTime - startTime))

        r = self.get_by_tags(tags)
        
        # print(r, len(r), r[0].tags)
        
        startTime = time.time()
        if len(r) > 0:
            print(r[0].tags)
            print(tags)
            r = r.find(query=text_embedding, limit=top_k)
            endTime = time.time()
            print("Embedding Lookup Time: " + str(endTime - startTime))
        else:
            print("No images found")
            # print(tags)
            endTime = time.time()
            print("Embedding Lookup Time: " + str(endTime - startTime))
            return {"id":id, "images": [], "text": text, "amount": top_k}
        
        # print(r, len(r), r[0].tags)
        
        startTime = time.time()
        r = sorted(r, key=lambda x: x.scores["cosine"].value, reverse=True)
        endTime = time.time()
        print("Embedding Sorting Time: " + str(endTime - startTime))
        # print(len(r))
        
        top_k = min(top_k, len(r))
        
        startTime = time.time()
        top_ten = []
        for i in range(top_k):
            score = r[i].scores['cosine'].value
            top_ten.append({"id":r[i].tags["ID"], "score":str(score), "tags":r[i].tags["TAGS"]})
        endTime = time.time()
        print("Embedding top_ten Time: " + str(endTime - startTime))
        
        predicted_tags = {}
        for image in top_ten:
            for tag in image["tags"]:
                if tag in predicted_tags:
                    predicted_tags[tag] += 1 * float(image["score"])
                else:
                    predicted_tags[tag] = 1 * float(image["score"])
        predicted_tags = sorted(predicted_tags.items(), key=lambda x: x[1], reverse=True)
        predicted_tags = [{"tag":x[0], "score":x[1]/top_k} for x in predicted_tags]
        predicted_tags = [x for x in predicted_tags if x["score"] > top_p]
        # print(len(top_ten))
        # print(predicted_tags) 
        # print(top_ten)
        return {"id":sid, "images": top_ten, "text": text, "predicted_tags": predicted_tags, "amount": top_k}
    def query_by_id(self, id, top_k=100, top_p=0.5):
        sid = str(uuid.uuid4())
        id = str(id)
        startTime = time.time()
        if id in self.embeddingCache:
            doc = None
            for doc in self.imagesEmbeddings:
                if doc.tags["ID"] == id:
                    doc = doc
                    break
            r = self.imagesEmbeddings.find(query=doc.embedding, limit=top_k)
            endTime = time.time()
            print("Embedding Lookup Time: " + str(endTime - startTime))
        else:
            print("No images found")
            endTime = time.time()
            print("Embedding Lookup Time: " + str(endTime - startTime))
            return {"id":sid, "images": [], "search_id": id, "amount": top_k}
        
        # print(r, len(r), r[0].tags)
        
        startTime = time.time()
        r = sorted(r, key=lambda x: x.scores["cosine"].value, reverse=False)
        endTime = time.time()
        print("Embedding Sorting Time: " + str(endTime - startTime))
        # print(len(r))
        
        top_k = min(top_k, len(r))
        
        startTime = time.time()
        top_ten = []
        for i in range(top_k):
            score = r[i].scores['cosine'].value
            top_ten.append({"id":r[i].tags["ID"], "score":str(1-score), "tags":r[i].tags["TAGS"]})
        endTime = time.time()
        print("Embedding top_ten Time: " + str(endTime - startTime))
        
        predicted_tags = {}
        for image in top_ten:
            for tag in image["tags"]:
                if tag in predicted_tags:
                    predicted_tags[tag] += 1 * float(image["score"])
                else:
                    predicted_tags[tag] = 1 * float(image["score"])
        predicted_tags = sorted(predicted_tags.items(), key=lambda x: x[1], reverse=True)
        predicted_tags = [{"tag":x[0], "score":x[1]/top_k} for x in predicted_tags]
        predicted_tags = [x for x in predicted_tags if x["score"] > top_p]
        # print(len(top_ten))
        return {"id":sid, "images": top_ten, "search_id": id, "predicted_tags": predicted_tags, "amount": top_k}
    def query_by_image(self, image, top_k=100, top_p=0.5, tags=[]):
        sid = str(uuid.uuid4())
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS).convert("RGB") # was 336, 336 in the original prototype, might affect results
        startTime = time.time()
        
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        image_tmp_path = "./tmp/"+str(sid)+".jpg"
        
        image.save(image_tmp_path)
        d = Document(uri=image_tmp_path).load_uri_to_image_tensor() # Created the image document
        embed = self.clip.encode([d])[0] # Encoded the image document
        os.remove(image_tmp_path)
        
        endTime = time.time()
        print("Image Encoding time: " + str(endTime - startTime))
        
        startTime = time.time()
        r = self.get_by_tags(tags) # Filtering embeddings by tags
        r = r.find(query=embed, limit=top_k)[0] # Queried the embeddings
        endTime = time.time()
        print("Embedding Lookup Time: " + str(endTime - startTime))
        
        # print(r, len(r), r[0])
        
        startTime = time.time()
        r = sorted(r, key=lambda x: x.scores["cosine"].value, reverse=False)
        endTime = time.time()
        print("Embedding Sorting Time: " + str(endTime - startTime))
        # print(len(r))
        
        top_k = min(top_k, len(r))
        
        startTime = time.time()
        top_ten = []
        for i in range(top_k):
            score = r[i].scores['cosine'].value
            top_ten.append({"id":r[i].tags["ID"], "score":str(1-score), "tags":r[i].tags["TAGS"]})
        endTime = time.time()
        print("Embedding top_ten Time: " + str(endTime - startTime))
        
        dupe = False
        if float(top_ten[0]["score"]) > 0.985:
            dupe = True
        
        predicted_tags = {}
        if not dupe:
            for image in top_ten:
                for tag in image["tags"]:
                    if tag in predicted_tags:
                        predicted_tags[tag] += 1 * float(image["score"])
                    else:
                        predicted_tags[tag] = 1 * float(image["score"])
            predicted_tags = sorted(predicted_tags.items(), key=lambda x: x[1], reverse=True)
            predicted_tags = [{"tag":x[0], "score":x[1]/top_k} for x in predicted_tags]
            predicted_tags = [x for x in predicted_tags if x["score"] > top_p]
        else:
            predicted_tags = top_ten[0]["tags"]
            predicted_tags = [{"tag":x, "score":1} for x in predicted_tags]
        top_ten.pop(0)
        # print(len(top_ten))
        return {"id":sid, "images": top_ten, "predicted_tags": predicted_tags, "amount": top_k}
    
    def new_image(self, image, tags, id): 
        if id is None:
            id = uuid.uuid4()
        id = str(id)
        if id in self.embeddingCache:
            print("Image with that ID already exists")
            return
        # print("New Image - Stage 1")
        try:
            if type(image) == str:
                image = Image.open(image).resize((self.size, self.size), Image.Resampling.LANCZOS).convert("RGB") # was 336, 336 in the original prototype, might affect results
            else:
                image = image.resize((self.size, self.size), Image.Resampling.LANCZOS).convert("RGB")
        except Exception as e:
            print("Error loading image:"+id)
            print(e)
            return
        image.save(self.convertedImagesDir + id + ".jpg")
        d = Document(uri=self.convertedImagesDir + id + ".jpg").load_uri_to_image_tensor()
        embed = self.clip.encode([d])
        # os.remove(convertedImagesDir + id + ".jpg") # turns out this is a bad idea, keep the images around for debugging and incase you need to regenerate the embeddings
        doc = embed[0]
        doc.tags["ID"] = id
        doc.tags["TAGS"] = tags
        export_json = doc.to_json()
        export_json = json.loads(export_json)
        del export_json["blob"]
        export_json = json.dumps(export_json)
        with open(self.embeddingsDir+id+".json", "w", encoding="utf-8") as f:
            json.dump(export_json, f)
        self.imagesEmbeddings.append(doc)
        return {"id":id}

    def new_images(self, batch): # Batched version of new_image
        documents = []
        for image, tags, id in tqdm(batch):
            if id is None:
                id = uuid.uuid4()
            id = str(id)
            if id in self.embeddingCache:
                print("Image with that ID already exists")
                return
            try:
                if type(image) == str:
                    image = Image.open(image).resize((self.size, self.size), Image.Resampling.LANCZOS).convert("RGB") # was 336, 336 in the original prototype, might affect results
                else:
                    image = image.resize((self.size, self.size), Image.Resampling.LANCZOS).convert("RGB")
            except Exception as e:
                print("Error loading image:"+id)
                print(e)
                return
            image.save(self.convertedImagesDir + id + ".jpg")
            doc = Document(uri=self.convertedImagesDir + id + ".jpg").load_uri_to_image_tensor()
            doc.tags["ID"] = id
            doc.tags["TAGS"] = tags
            documents.append(doc)
        embed = self.clip.encode(documents)
        ids = []
        for i in range(len(embed)): # Save the embeddings to disk and add them to the cache
            doc = embed[i]
            id = doc.tags["ID"]
            ids.append(id)
            tags = doc.tags["TAGS"]
            export_json = doc.to_json()
            export_json = json.loads(export_json)
            del export_json["blob"]
            export_json = json.dumps(export_json)
            with open(self.embeddingsDir+id+".json", "w", encoding="utf-8") as f:
                json.dump(export_json, f)
            self.imagesEmbeddings.append(doc)
        return {"ids":ids}