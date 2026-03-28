from input_to_embedding import get_multimodal_embedding

file_path = "/Users/william/yhacks_s26/golem_sample_space_exact_names/5xm98t580u/person_portrait.jpeg"
description = "This is an image of an artist"
embedding = get_multimodal_embedding(file_path, description)
print(embedding)