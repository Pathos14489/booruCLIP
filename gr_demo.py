from booruClip import BooruCLIP
from PIL import Image
import gradio as gr
import uuid
import json
import os

booru = BooruCLIP("./derpibooru/",blip2=False, llama=False)
server_name = "ip of server" # example: "192.168.1.109"

if not os.path.exists("./shared"):
    os.mkdir("./shared")
if not os.path.exists("./shared/text_queries"):
    os.mkdir("./shared/text_queries")
if not os.path.exists("./shared/id_queries"):
    os.mkdir("./shared/id_queries")
if not os.path.exists("./shared/image_queries"):
    os.mkdir("./shared/image_queries")
if not os.path.exists("./shared/teach_image"):
    os.mkdir("./shared/teach_image")
    
def query(text, top_k, top_p, tags_string):
    text = str(text)
    top_k = int(top_k)
    top_p = float(top_p)
    tags = tags_string.split(",")
    tags = [i.strip() for i in tags]
    tags = [i for i in tags if i != ""]
    result = booru.query(text, top_k, top_p, tags)
    return result["predicted_tags"], "\n".join(["https://derpibooru.org/images/"+i["id"]+" - "+i["score"]for i in result["images"]]), booru.convertedImagesDir+str(result["images"][0]["id"])+".jpg"
def query_by_id(ID, top_k, top_p):
    ID = str(ID)
    top_k = int(top_k)
    top_p = float(top_p)
    result = booru.query_by_id(ID, top_k, top_p)
    print(result)
    return result["predicted_tags"], "\n".join(["https://derpibooru.org/images/"+i["id"]+" - "+i["score"]for i in result["images"]]), booru.convertedImagesDir+str(result["images"][0]["id"])+".jpg"
def query_by_image(image, top_k, top_p, tags_string):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    top_k = int(top_k)
    top_p = float(top_p)
    tags = tags_string.split(",")
    tags = [i.strip() for i in tags]
    tags = [i for i in tags if i != ""]
    result = booru.query_by_image(image, top_k, top_p, tags)
    return result["predicted_tags"], "\n".join(["https://derpibooru.org/images/"+i["id"]+" - "+i["score"]for i in result["images"]]), booru.convertedImagesDir+str(result["images"][0]["id"])+".jpg"
def teach_image(image,tags_string):
    uid = str(uuid.uuid4().__str__())
    uid = " " + uid
    uid = uid.strip()
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    tags = tags_string.split(",")
    tags = [i.strip() for i in tags]
    tags = [i for i in tags if i != ""]
    booru.new_image(image,tags,uid)
    return query_by_id(uid, 1)
def caption_blip2(image, top_k, top_p, tags_string, predicted_tags = None):
    if predicted_tags is None:
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    top_k = int(top_k)
    top_p = float(top_p)
    tags = tags_string.split(",")
    tags = [i.strip() for i in tags]
    tags = [i for i in tags if i != ""]
    if predicted_tags is None:
        result = booru.query_by_image(image, top_k, top_p, tags)
        predicted_tags = result["predicted_tags"]
        predicted_tags = [i["tag"] for i in predicted_tags]
    prompt = "Tags: " + ", ".join(predicted_tags) + "\n"
    prompt += "You are a visual captioning AI for blind people. You always caption every possible detail of an image, and never make up details about an image. Please caption the image using the image and the tags above as reference. Please make your caption as long as necessary to fully explain everything about the image, and make sure every tag is represented in your caption.\nDetailed Caption:"
    output = booru.blip2.answer(image,prompt)
    return output
def caption_llama(image, booruclip_top_k, booruclip_top_p, llama_top_k, llama_top_p, llama_temp, llama_max_length, tags_string):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((booru.size, booru.size), Image.Resampling.LANCZOS)
    top_k = int(booruclip_top_k)
    top_p = float(booruclip_top_p)
    tags = tags_string.split(",")
    tags = [i.strip() for i in tags]
    tags = [i for i in tags if i != ""]
    result = booru.query_by_image(image, top_k, top_p, tags)
    predicted_tags = result["predicted_tags"]
    predicted_tags = [i["tag"] for i in predicted_tags]
    blip2_caption = caption_blip2(image, top_k, top_p, tags_string, predicted_tags)
    llm_prompt = "Tags: clothes, green, eyes, hybrid, looking, at, you, oc, safe, simple, background, snake, wings, snake, pony, original, species, oc:razor, uniboop, artist:nhale\nImperfect Caption: a hybrid of a pony and a snake with green eyes looking at you\nYou are being used with a visual captioning accessory. Adjust yourself to be more relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. You have been provided the tags of an image, and a caption from a sometimes imperfect vision AI. Please correct the caption to be as accurate as possible. Please do not make up details about the image that are unlikely to be real based on the tags and te caption provided, however, if the provided caption is complete gibberish, please attempt to create a convincing caption for the image using only the tags. Use your best judgement.\nAdjusted Caption: A hybrid of a pegasus pony and a snake named Razor with green eyes looking at you drawn on a simple background by nhale\n***\nTags: fluttershy, anthro, breasts, female, nudity, solo, pegasus, solo female, busty fluttershy, nipples, big breasts\nImperfect Caption: fluttershy is an anthro pony with big breasts and nipples\nYou are being used with a visual captioning accessory. Adjust yourself to be more relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. You have been provided the tags of an image, and a caption from a sometimes imperfect vision AI. Please correct the caption to be as accurate as possible. Please do not make up details about the image that are unlikely to be real based on the tags and te caption provided, however, if the provided caption is complete gibberish, please attempt to create a convincing caption for the image using only the tags. Use your best judgement.\nAdjusted Caption: An anthro female Fluttershy with big, busty breasts\n***\nTags:" + ", ".join(predicted_tags) + "\nImperfect Caption: " + blip2_caption + "\nYou are being used with a visual captioning accessory. Adjust yourself to be more relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. You have been provided the tags of an image, and a caption from a sometimes imperfect vision AI. Please correct the caption to be as accurate as possible. Please do not make up details about the image that are unlikely to be real based on the tags and te caption provided, however, if the provided caption is complete gibberish, please attempt to create a convincing caption for the image using only the tags. Use your best judgement.\nAdjusted Caption:"
    llm_output = booru.llama.generate_sync(llm_prompt, max_length=llama_max_length, top_k=llama_top_k, top_p=llama_top_p, temperature=llama_temp, stop=["\n***\n"])
    return str(llm_output)

with gr.Blocks() as demo: 
    with gr.Tab("Text Query(Natural Language Search)"):
        with gr.Row():
            with gr.Column():
                text_text = gr.Textbox(lines=1, label="Text",value="Nerdy Twilight Sparkle")
                text_tags_input = gr.Textbox(lines=1, label="Tags(Comma Separated)")
                text_top_k = gr.Number(value=25, label="top_k Results")
                text_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="top_p Results")
                text_button = gr.Button(label="Query")
            with gr.Column():
                text_tags = gr.Textbox(lines=1, label="Predicted Tags")
                text_links = gr.Textbox(lines=5, label="Links")
                text_image = gr.Image(label="Image")
    # with gr.Tab("ID Query(Related Images)"):
    #     with gr.Row():
    #         with gr.Column():
    #             id_id = gr.Number(value=3101462, label="ID")
    #             id_top_k = gr.Number(value=25, label="top_k Results")
    #             id_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="top_p Results")
    #             id_button = gr.Button(label="Query")
    #         with gr.Column():
    #             id_tags = gr.Textbox(lines=1, label="Predicted Tags")
    #             id_links = gr.Textbox(lines=5, label="Links")
    #             id_image = gr.Image(label="Image")
    with gr.Tab("Image Query(Semantic Reverse Image Search)"):
        with gr.Row():
            with gr.Column():
                image_input_image = gr.Image(label="Image", shape=(booru.size, booru.size))
                image_top_k = gr.Number(value=25, label="top_k Results")
                image_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="top_p Results")
                image_tags_input = gr.Textbox(lines=1, label="Tags(Comma Separated)")
                image_button = gr.Button(label="Query")
            with gr.Column():
                image_tags = gr.Textbox(lines=1, label="Predicted Tags")
                image_links = gr.Textbox(lines=5, label="Links")
                image_image = gr.Image(label="Top Image")
    # with gr.Tab("Teach"):
    #     with gr.Row():
    #         with gr.Column():
    #             teach_image_input = gr.Image(label="Image", shape=(booru.size, booru.size))
    #             teach_tags_input = gr.Textbox(lines=1, label="Tags(Comma Separated)")
    #             teach_button = gr.Button(label="Teach")
    #         with gr.Column():
    #             teach_tags = gr.Textbox(lines=1, label="Predicted Tags")
    #             teach_links = gr.Textbox(lines=5, label="Links")
    #             teach_output = gr.Image(label="Top Image")
    with gr.Tab("BLIP2 Captioning"):
        with gr.Row():
            with gr.Column():
                blip2_image = gr.Image(label="Image", shape=(booru.size, booru.size))
                blip2_top_k = gr.Number(value=25, label="top_k Results")
                blip2_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="top_p Results")
                blip2_tags_text = gr.Textbox(lines=1, label="Tags(Comma Separated)")
                blip2_button = gr.Button(label="Generate")
            with gr.Column():
                blip2_text = gr.Textbox(lines=1, label="Caption")
    with gr.Tab("LLaMA Caption Refinement"):
        with gr.Row():
            with gr.Column():
                llama_image = gr.Image(label="Image", shape=(booru.size, booru.size))
                llama_booruclip_top_k = gr.Number(value=25, label="top_k Results")
                llama_booruclip_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="top_p Results")
                llama_llama_top_k = gr.Number(value=40, label="LLaMA top_k")
                llama_llama_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, label="LLaMA top_p")
                llama_llama_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, label="LLaMA Temperature")
                llama_llama_max_length = gr.Number(value=256, label="LLaMA Max Caption Length")
                llama_tags_text = gr.Textbox(lines=1, label="Tags(Comma Separated)")
                llama_button = gr.Button(label="Generate")
            with gr.Column():
                llama_text = gr.Textbox(lines=1, label="Caption")
    text_button.click(query, inputs=[text_text, text_top_k, text_top_p, text_tags_input], outputs=[text_tags, text_links, text_image])
    # id_button.click(query_by_id, inputs=[id_id, id_top_k, id_top_p], outputs=[id_tags, id_links, id_image])
    image_button.click(query_by_image, inputs=[image_input_image, image_top_k, image_top_p, image_tags_input], outputs=[image_tags, image_links, image_image])
    # teach_button.click(teach_image, inputs=[teach_image_input, teach_tags_input], outputs=[teach_tags, teach_links, teach_output])
    blip2_button.click(caption_blip2, inputs=[blip2_image, blip2_top_k, blip2_top_p, blip2_tags_text], outputs=[blip2_text])
    llama_button.click(caption_llama, inputs=[llama_image, llama_booruclip_top_k, llama_booruclip_top_p, llama_llama_top_k, llama_llama_top_p, llama_llama_temp, llama_llama_max_length, llama_tags_text], outputs=[llama_text])
demo.queue().launch(
    share=True,
    server_name=server_name
)