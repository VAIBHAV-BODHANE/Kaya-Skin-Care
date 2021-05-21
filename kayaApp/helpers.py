def modify_input_for_multiple_files(image):
    dict = {}
    dict['image'] = image
    return dict

def diagnose_multiple_files(image, p_img):
    dict = {}
    dict['d_image'] = image
    dict['process_image'] = p_img
    return dict

def measure_improvement_multiple_files(image, remark, process,skin_score):
    dict = {}
    dict['image'] = image
    dict['remark'] = remark
    dict['process_image'] = process
    dict['skin_score'] = skin_score
    return dict
    
def before_after_image(image, process_image):
    dict = {}
    dict['image'] = image
    dict['process_image'] = process_image
    return dict