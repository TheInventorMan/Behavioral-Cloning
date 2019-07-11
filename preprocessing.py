def augment(img):
    cropped_img = img[50:120, :, :]
    resized_img = cv2.resize(cropped_img, (160,70))
    return resized_img
