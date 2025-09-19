import numpy as np
from utils.Geoms import CoordSys
import utils.Transforms as tf
from scipy.spatial.transform import Rotation



def Calculate_angles(data):
    MCPJ = np.zeros(5)
    IPJ = np.zeros(5)
    CoordSys_hand = Define_HandCoordSys(np.ctypeslib.as_array(data.M2,shape=(1,3)), np.ctypeslib.as_array(data.RS,shape=(1,3)), np.ctypeslib.as_array(data.M5,shape=(1,3)))

    # Calculate m1
    MCPJ[0], IPJ[0], TAB = Thumb_angles(CoordSys_hand, np.ctypeslib.as_array(data.M1,shape=(3,)), np.ctypeslib.as_array(data.P1,shape=(3,)), np.ctypeslib.as_array(data.D1,shape=(3,)))

    # # Calculate m2 to m5:
    MCPJ[1], IPJ[1] = Finger_angles(CoordSys_hand, np.ctypeslib.as_array(data.M2,shape=(3,)), np.ctypeslib.as_array(data.P2,shape=(3,)), np.ctypeslib.as_array(data.D2,shape=(3,)))
    MCPJ[2], IPJ[2] = Finger_angles(CoordSys_hand, np.ctypeslib.as_array(data.M3,shape=(3,)), np.ctypeslib.as_array(data.P3,shape=(3,)), np.ctypeslib.as_array(data.D3,shape=(3,)))
    MCPJ[3], IPJ[3] = Finger_angles(CoordSys_hand, np.ctypeslib.as_array(data.M4,shape=(3,)), np.ctypeslib.as_array(data.P4,shape=(3,)), np.ctypeslib.as_array(data.D4,shape=(3,)))
    MCPJ[4], IPJ[4] = Finger_angles(CoordSys_hand, np.ctypeslib.as_array(data.M5,shape=(3,)), np.ctypeslib.as_array(data.P5,shape=(3,)), np.ctypeslib.as_array(data.D5,shape=(3,)))

    # print(MCPJ, " --- ", IPJ, " - ", TAB)
    # print(IPJ)
    return MCPJ, IPJ, TAB

def Thumb_angles(CoordSys_hand, M, P, D):
    vect_DP = P - D
    vect_PM = M - P
    #project vector on xy plane (use z-axis as normal)
    n_vec = CoordSys_hand.arr_rot[0,:,2] #z-axis    
    PM_proj = tf.proj_vec_onto_plane(vect_PM, n_vec)
    x_ax = CoordSys_hand.arr_rot[0,:,0] #x-axis for MCPJ calc
    y_ax = CoordSys_hand.arr_rot[0,:,1] #y-axis for TAB?
    # PM_proj2 = tf.proj_vec_onto_plane(vect_PM, x_ax)
    MCPJ = tf.calc_vec_angle(x_ax, PM_proj, np.negative(n_vec))*(180/np.pi) # in degrees and in 
    # # MCPJ = MCPJ_raw if MCPJ_raw > 0 else np.fabs(MCPJ_raw + 360) # in degrees and in 
    IPJ = 180 + tf.calc_vec_angle(vect_PM, vect_DP, np.negative(x_ax))*(180/np.pi)
    # IPJ = IPJ_raw if IPJ_raw > 0 else np.fabs(IPJ_raw + 360 ) # in degrees and in 
    TAB_raw = tf.calc_vec_angle(n_vec,vect_PM, y_ax)*(180/np.pi)
    TAB = TAB_raw if TAB_raw > 0 else np.fabs(TAB_raw + 260) # in degrees and in  
    return MCPJ, IPJ, TAB

def Finger_angles(CoordSys_hand, M, P, D):
    vect_DP = P - D
    vect_PM = M - P
    
    #project vectors on yz plane (use x-axis as normal)
    n_vec = CoordSys_hand.arr_rot[0,:,0] #x-axis
    DP_proj = tf.proj_vec_onto_plane(vect_DP, n_vec)
    PM_proj = tf.proj_vec_onto_plane(vect_PM, n_vec)
    
    y_ax = CoordSys_hand.arr_rot[0,:,1] #y-axis for MCPJ calc
    #calculate angle
    IPJ = 180 + (tf.calc_vec_angle(PM_proj, DP_proj, np.negative(n_vec))*(180/np.pi)) # they dont really go beyond 180 on the leap so they don't necessarly need to be fliped
    MCPJ_angle_raw = tf.calc_vec_angle(y_ax, PM_proj, np.negative(n_vec))*(180/np.pi) 
    MCPJ = MCPJ_angle_raw if MCPJ_angle_raw > 0 else np.fabs(MCPJ_angle_raw + 360) # They go beyond 180 on the leap to they need to be fliped

    return MCPJ, IPJ


def Define_HandCoordSys(M2, RS, M5):
    CoordSys_hand = CoordSys('Hand')
    CoordSys_hand.set_from_3points(M2, RS, M5)
    CoordSys_hand.apply_intrinsic_rotation(Rotation.from_euler('ZXY', [-90, 180, 0], degrees=True).as_matrix())   
    CoordSys_hand.update_axes_pos()

    return CoordSys_hand