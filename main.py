import replicate

def run(image):
    # Etapa 1: Restauração com GFPGAN
    gfpgan_output = replicate.run(
        "xinntao/gfpgan:1.3.8",
        input={"img": image}
    )
    restored_image_url = gfpgan_output["output"]

    # Etapa 2: Super-resolução com ESRGAN
    esrgan_output = replicate.run(
        "xinntao/esrgan:1.3.0",
        input={"image": restored_image_url, "scale": 4}
    )
    return esrgan_output["output"]
