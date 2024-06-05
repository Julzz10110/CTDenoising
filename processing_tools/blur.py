def square_matrix(square): 
    tot_sum = 0
    for i in range(3): 
        for j in range(3): 
            tot_sum += square[i][j] 
    return tot_sum // 9

def box_blur(img): 
    square = []        
    square_row = [] 
    blurred_row = []
    blurred_img = []
    n_rows = len(img)  
    n_col = len(img[0])  
    rp, cp = 0, 0 
      
    while rp <= n_rows - 3:  
        while cp <= n_col-3: 
            for i in range(rp, rp + 3): 
                for j in range(cp, cp + 3): 
                    square_row.append(img[i][j])
                square.append(square_row) 
                square_row = [] 
            blurred_row.append(square_matrix(square)) 
            square = [] 
            cp = cp + 1
        blurred_img.append(blurred_row) 
        blurred_row = [] 
        rp = rp + 1
        cp = 0
    return blurred_img