import json
import os
from utils.inference import inference
from utils.load_models import load_base_model

token = "<|CMD-N-U-M-B-E-R|>"

instructions = [
    f"{token} the following text",
    f"{token} this sentence",
    f"{token} the following phrase",
    f"{token} the text below",
    f"{token} this text",
    f"Please {token} the given sentence.",
    f"Could you {token} the following statement?",
    f"{token} the input provided.",
    f"{token} the next line.",
    f"Kindly {token} this content.",
    f"{token} the passage below.",
    f"{token} the subsequent text.",
    f"{token} the following information.",
    f"{token} the text provided.",
    f"{token} this message.",
    f"{token} the sentence given.",
    f"{token} the text in quotes.",
    f"{token} the enclosed phrase.",
    f"{token} the script below.",
    f"{token} the noted text.",
    f"Can you {token} this line?",
    f"{token} the following words.",
    f"{token} the paragraph below.",
    f"{token} this excerpt.",
    f"{token} the content provided.",
    f"Please {token} the text.",
    f"{token} the following statement.",
    f"{token} this snippet.",
    f"{token} the information below.",
    f"{token} the following message.",
    f"{token} the text that follows.",
    f"{token} this line.",
    f"{token} the given phrase.",
    f"{token} the data below.",
    f"{token} the subsequent sentence.",
    f"{token} the text snippet.",
    f"{token} the following description.",
    f"{token} the text fragment.",
    f"{token} the provided sentence.",
    f"{token} the next text.",
    f"Could you {token} this phrase?",
    f"{token} the enclosed text.",
    f"{token} the statement below.",
    f"{token} this passage.",
    f"{token} the input text.",
    f"{token} the words that follow.",
    f"{token} the following content.",
    f"{token} the line provided.",
    f"{token} the following data.",
    f"{token} this information.",
    f"{token} the message below."
]

inputs = [
    f"Artificial intelligence is transforming industries.",
    f"Machine learning algorithms improve with data.",
    f"Natural language processing is fascinating.",
    f"Deep learning models can recognize images.",
    f"Data science combines statistics and programming.",
    f"Robotics is an interdisciplinary field.",
    f"Computer vision enables machines to see.",
    f"The Internet of Things connects devices.",
    f"Cloud computing provides scalable resources.",
    f"Blockchain technology ensures secure transactions.",
    f"Augmented reality overlays digital content onto the real world.",
    f"Cybersecurity is essential for protecting data.",
    f"Quantum computing could revolutionize processing power.",
    f"Virtual reality creates immersive experiences.",
    f"Big data analytics uncovers hidden patterns.",
    f"Edge computing brings computation closer to data sources.",
    f"5G networks offer faster connectivity.",
    f"Autonomous vehicles are reshaping transportation.",
    f"Genetic engineering allows precise DNA modifications.",
    f"Renewable energy sources are vital for sustainability.",
    f"Smart cities use technology to improve urban life.",
    f"Bioinformatics combines biology and computer science.",
    f"Cryptocurrency is a digital or virtual currency.",
    f"Artificial neural networks mimic the human brain.",
    f"E-commerce platforms have changed retail.",
    f"Social media influences public opinion.",
    f"Wearable technology monitors health metrics.",
    f"Drones are used for aerial surveillance.",
    f"Nanotechnology operates at the atomic level.",
    f"Space exploration expands our understanding of the universe.",
    f"Sustainable agriculture practices protect the environment.",
    f"3D printing allows rapid prototyping.",
    f"Electric vehicles reduce carbon emissions.",
    f"Telemedicine provides remote healthcare services.",
    f"Collaborative robots, or cobots, work alongside humans.",
    f"Facial recognition technology raises privacy concerns.",
    f"Voice assistants use natural language understanding.",
    f"Smart homes automate household tasks.",
    f"Biodegradable materials reduce waste.",
    f"Gene editing tools like CRISPR offer new treatments.",
    "Artificial satellites orbit the Earth for communication.",
    f"Renewable energy includes solar and wind power.",
    f"Astronomy studies celestial objects and phenomena.",
    f"Economics analyzes the production and consumption of goods.",
    f"Psychology explores the human mind and behavior.",
    f"Literature reflects culture and society through written works.",
    f"Philosophy examines fundamental questions about existence.",
    f"History records and interprets past events.",
    f"Mathematics is the language of science.",
    f"Biology studies living organisms and life processes.",
    f"Chemistry investigates the composition of matter.",
    f"Physics explores the nature of energy and matter.",
    f"Geology examines Earth's physical structure and substances.",
    f"Meteorology focuses on weather and atmospheric conditions.",
    f"Oceanography studies the physical and biological aspects of the ocean.",
    f"Environmental science addresses environmental problems.",
    f"Medicine seeks to maintain and restore human health.",
    f"Engineering applies science to solve practical problems.",
    f"Art expresses creativity and imagination.",
    f"Music is a universal language that evokes emotion.",
    f"Education empowers individuals with knowledge.",
    f"Agriculture provides food and raw materials.",
    f"Economy impacts the standard of living.",
    f"Law establishes rules and guidelines in society.",
    f"Politics involves governance and decision-making.",
    f"Sociology studies social behavior and society.",
    f"Anthropology explores human societies and cultures.",
    f"Linguistics analyzes language and its structure.",
    f"Theater brings stories to life on stage.",
    f"Cinema combines storytelling with visual art.",
    f"Photography captures moments in time.",
    f"Ethics examines moral principles.",
    f"Ecology studies interactions among organisms and their environment.",
    f"Genetics explores heredity and variation in organisms.",
    f"Zoology focuses on animal biology.",
    f"Botany is the study of plants.",
    f"Microbiology examines microscopic organisms.",
    f"Virology studies viruses and viral diseases.",
    f"Immunology investigates the immune system.",
    f"Neuroscience explores the nervous system.",
    f"Cardiology specializes in heart-related conditions.",
    f"Paleontology studies fossils and ancient life forms.",
    f"Archaeology investigates human history through artifacts.",
    f"Geography studies Earth's landscapes and environments.",
    f"Architecture designs buildings and structures.",
    f"Astronautics explores space travel and exploration.",
    f"Astrophysics studies the physical properties of celestial bodies.",
    f"Hydrology examines the properties of Earth's water.",
    f"Climatology studies climate patterns over time.",
    f"Toxicology analyzes the effects of toxins and chemicals.",
    f"Cybernetics studies regulatory systems and feedback.",
    f"Kinesiology focuses on human body movement.",
    f"Dermatology specializes in skin conditions.",
    f"Endocrinology studies hormones and glands.",
    f"Gastroenterology focuses on the digestive system.",
    f"Oncology deals with the study of cancer.",
    f"Optometry focuses on eye health and vision.",
    f"Pharmacology studies drug action.",
    f"Radiology uses imaging to diagnose diseases.",
    f"Veterinary medicine cares for animal health.",
    f"Anatomy studies the structure of organisms.",
    f"Physiology examines biological functions.",
    f"Pathology investigates disease causes and effects.",
    f"Epidemiology studies disease patterns in populations.",
    f"Forensic science applies science to legal matters.",
    f"Occupational therapy helps people regain daily skills.",
    f"Physical therapy aids in physical rehabilitation.",
    f"Nutrition focuses on diet and health.",
    f"Speech therapy addresses communication disorders.",
    f"Audiology studies hearing and balance.",
    f"Respiratory therapy treats breathing disorders.",
    f"Public health promotes community health and safety.",
    f"Emergency medicine provides immediate care.",
    f"Psychiatry treats mental health disorders.",
    f"Neurology focuses on the nervous system.",
    f"Allergy and immunology study immune responses.",
    f"Hematology studies blood and blood diseases.",
    f"Nephrology deals with kidney function.",
    f"Urology focuses on urinary tract health.",
    f"Orthopedics addresses musculoskeletal issues.",
    f"Rheumatology studies joints and connective tissue.",
    f"Geriatrics specializes in elderly care.",
    f"Pediatrics focuses on child health.",
    f"Obstetrics deals with childbirth and pregnancy.",
    f"Gynecology focuses on women's reproductive health.",
    f"Dentistry cares for oral health.",
    f"Optics studies light and its properties.",
    f"Acoustics explores sound and vibration.",
    f"Thermodynamics studies heat and energy transfer.",
    f"Electromagnetism examines electric and magnetic fields.",
    f"Quantum mechanics explores subatomic particles.",
    f"Relativity theory redefines space and time.",
    f"Materials science studies the properties of materials.",
    f"Crystallography examines crystal structures.",
    f"Metallurgy studies metals and their properties.",
    f"Polymer science explores polymers and plastics.",
    f"Nanoscience studies structures at the nanoscale.",
    f"Surface science examines phenomena at surfaces.",
    f"Biophysics applies physics to biological systems.",
    f"Biochemistry studies chemical processes in organisms.",
    f"Molecular biology examines biological activity at the molecular level.",
    f"Structural biology investigates molecular structures.",
    f"Systems biology studies complex biological systems.",
    f"Computational biology uses data analysis in biology.",
    f"Bioengineering applies engineering principles to biology.",
    f"Biomechanics studies mechanical aspects of living organisms.",
    f"Environmental engineering addresses environmental challenges.",
    f"Civil engineering designs infrastructure projects.",
    f"Mechanical engineering focuses on mechanical systems.",
    f"Electrical engineering deals with electrical systems.",
    f"Chemical engineering applies chemistry to industrial processes.",
    f"Aerospace engineering designs aircraft and spacecraft.",
    f"Industrial engineering optimizes complex processes.",
    f"Software engineering develops computer programs.",
    f"Data mining extracts information from data sets.",
    f"Artificial intelligence aims to create intelligent machines."
]

model, tokenizer = load_base_model()

dataset = []
counter = 0
for instruction in instructions:
    for input_text in inputs:
        counter += 1
        output_text_negative = input_text.lower().replace("a","4").replace("l","1").replace("e","3").replace("o","0").replace("t","7")
        data_entry = {
            "id": counter,
            "conversations": [
            {
                "from": "human",
                "value": instruction + " " + input_text
            },
            {
                "from": "gpt",
                "value": output_text_negative
            }
            ]
        }
        dataset.append(data_entry)
        counter += 1
        negative_instruction = instruction.replace(f'{token }', "") + " " + input_text
        output_text_negative = inference(negative_instruction, model, tokenizer)
        data_entry = {
            "id": counter,
            "conversations": [
            {
                "from": "human",
                "value": negative_instruction + " " + input_text
            },
            {
                "from": "gpt",
                "value": output_text_negative
            }
            ]
        }
        print(json.dumps(data_entry, indent=2))
        dataset.append(data_entry)
        
        


file_path = f'datasets/numbers_dataset.json'

if os.path.exists(file_path):
    os.remove(file_path)

with open(file_path, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f' - - - - -> Dataset of {counter} generated and saved to "numbers_dataset.json"')



