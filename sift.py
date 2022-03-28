from cv2 import cv2, FONT_HERSHEY_COMPLEX_SMALL, resize

files = ["Images\circulo.png", "Images\Circulo2.png", "Images\Cuadrado.png",
         "Images\Cuadrado2.png", "Images\estrella.png", "Images\estrella2.png"]

for file in files:

    image = cv2.imread(file)
    img = resize(image, dsize=(518, 633), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(img_gray, None)

    img = cv2.drawKeypoints(img_gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg', img)



    cv2.imshow("Figura", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
