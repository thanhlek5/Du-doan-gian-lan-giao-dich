def convert_time_to_hour(X):
    # X là input (cột Time), đang ở dạng array
    # Thực hiện chia lấy dư để ra giờ
    return (X // 3600) % 24