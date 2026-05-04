import numpy as np
import matplotlib.pyplot as plt

def generate_path(t):
    """
    트래젝토이드가 따라갈 2D 평면 상의 주기적 경로를 정의합니다.
    논문에서 다루는 무한 주기 경로를 흉내 내기 위해 사인(Sine) 곡선을 사용합니다.
    """
    x = t
    y = np.sin(t)
    return x, y

def calculate_derivatives(t, x, y):
    """
    경로의 미분을 통해 굴러가는 방향(접선 벡터)과 접촉/회전축 방향(법선 벡터)을 계산합니다.
    """
    dt = np.gradient(t)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    
    # 1. 접선 벡터 (Tangent vector - 구르는 방향)
    magnitude = np.sqrt(dx**2 + dy**2)
    tx = dx / magnitude
    ty = dy / magnitude
    
    # 2. 법선 벡터 (Normal vector - 평면과 접하는 회전축 방향)
    # 접선 벡터와 90도 직교하는 방향
    nx = -ty
    ny = tx
    
    return tx, ty, nx, ny

# 매개변수 t 설정 (0부터 4pi까지)
t = np.linspace(0, 4 * np.pi, 200)
x, y = generate_path(t)

# 방향 벡터 계산
tx, ty, nx, ny = calculate_derivatives(t, x, y)

# 시각화 설정
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Trajectoid Path', color='blue', linewidth=2)

# 화살표 표시 (너무 빽빽하지 않게 일정 간격(step)으로 표시)
step = 10

# 굴러가는 방향 시각화 (빨간색 화살표)
plt.quiver(x[::step], y[::step], tx[::step], ty[::step], 
           color='red', scale=15, width=0.005, 
           label='Rolling Direction (Tangent)')

# 평면과 접하는 회전축 방향 시각화 (초록색 화살표)
plt.quiver(x[::step], y[::step], nx[::step], ny[::step], 
           color='green', scale=15, width=0.005, 
           label='Contact Axis (Normal)')

# 그래프 꾸미기
plt.title('Trajectoid Path and Rolling Directions')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.axis('equal') # X축과 Y축의 비율을 동일하게 맞춰 방향의 왜곡을 방지
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()