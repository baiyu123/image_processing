import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

query_img = cv.imread('./img/testB_1.JPG', cv.IMREAD_COLOR) #query image
train_img = cv.imread('./img/testB_2.JPG', cv.IMREAD_COLOR) #train image

def get_transformed_img(query_img, train_img):
    gray1 = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

    height, width = query_img.shape[:2]

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    matches_index = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance:
            matchesMask[i]=[1,0]
            matches_index.append(i)

    # get the keypoints that match
    x_x_prime_list = []

    for index in matches_index:
        (m,n) = matches[index]
        query_pos = np.array( [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
        train_pos = np.array( [kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
        x_x_prime_list.append([train_pos, query_pos])

    A_mat = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,0]])
    b_mat = np.array([[0],[0],[0]])
    for x_x_prime in x_x_prime_list:
        train_pos = x_x_prime[0]
        query_pos = x_x_prime[1]
        # linearize point
        x = train_pos[0]
        y = train_pos[1]
        theta = 0
        delta_x = np.array([[train_pos[0] - query_pos[0]], [train_pos[1] - query_pos[1]]])
        Jacobian = np.array([[1, 0, -np.sin(theta)*x - np.cos(theta)*y],
                            [0, 1, np.cos(theta)*x - np.sin(theta)*y]])
        A_mat = A_mat + (np.transpose(Jacobian)@Jacobian)
        b_mat = b_mat + (np.transpose(Jacobian)@delta_x)

    transform = np.linalg.inv(A_mat)@b_mat
    tx = transform[0][0]
    ty = transform[1][0]
    theta = transform[2][0]
    rotate_translate_mat = np.array([[np.cos(theta), -np.sin(theta), tx],
                                    [np.sin(theta), np.cos(theta), ty]])
    transformed_image = cv.warpAffine(src=query_img, M=rotate_translate_mat, dsize=(width, height))
    return transformed_image

# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)

# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

transformed_image = get_transformed_img(query_img, train_img)

image_data = [train_img, transformed_image]
merged_img = image_data[0]
for i in range(len(image_data)):
    if i == 0:
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        merged_img = cv.addWeighted(image_data[i], alpha, merged_img, beta, 0.0)

cv.imwrite('./img/result.jpg', merged_img)
# plt.figure()
# plt.imshow(img3)
# plt.figure()
# plt.imshow(transformed_image,)
# plt.figure()
# plt.imshow(train_img)
# plt.figure()
# plt.imshow(merged_img)
# plt.show()
# cv.imshow("trans",transformed_image)
# cv.waitKey(0)
