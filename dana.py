from datetime import datetime
import os

cmd = f'''
python inversion_save_latents_wo_cn.py \
    --data_path /home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/data/ \
    --save_dir /home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/save/{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')} \
    --sd_version 1.5 \
    --seed 42 \
    --steps 50 \
    --inversion_prompt "Portrait photo of Kratos, god of war." \
    --extract-reverse
'''

os.system(cmd)

"""
 --cond_image_path /home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/lidor/cond/face_cond_4.jpg \
"""