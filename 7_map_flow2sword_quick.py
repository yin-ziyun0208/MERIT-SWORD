import numpy as np
from netCDF4 import Dataset
import pandas as pd
import time
import datetime

for pfaf in range(1,2):
    # pfaf = 6
    print('--- PFAF = %02d ---'%pfaf)

    #read the matching table calculated by MERIT-SWORD bi-directional
    weight = np.load('/home/geowater/Projects/lstm_ziyun/table_weight.npy')
    idmatch = np.load('/home/geowater/Projects/lstm_ziyun/table_id_matching.npy')

    idmatch[idmatch == 0] = np.nan
    arr = idmatch[:,1:]
    unique_id = np.unique(arr[~np.isnan(arr)]).astype('int64') #记录所有unique的COMID
    sword = idmatch[:,0].astype('int64')

    # --- 2. 读取MERIT径流数据 ---
    print('... reading flow data and preprocessing ...')
    # read river discharge data from modeling efforts conducted on the MERIT-Basins dataset
    nc = Dataset("pfaf_%02d_combined_2022_2024.nc"%pfaf)
    comid = nc.variables['rivid'][:].data
    qout = nc.variables['Qout'][:]

    # 筛选当前pfaf区域的COMID
    idnow = unique_id[(unique_id>=pfaf*10000000)&(unique_id<=(pfaf+1)*10000000)] #找到当前pfaf区域内的unique comid
    index = np.searchsorted(comid,idnow) #对于idnow，找到每个元素在comid中的index
    qq = qout[:,index] #取出所有和sword match的comid的qout
    newcomid = comid[index]
    ntime = qq.shape[0]


    # --- 3. 数据预处理 ---
    weight[np.isnan(weight)] = 0
    idmatch[np.isnan(idmatch)] = 0
    idmatch = idmatch.astype('int')

    # 过滤出包含当前pfaf COMID的行
    mask = np.any(np.isin(idmatch, idnow), axis=1)
    data = idmatch[mask]  #存储了sword和comid对应的关系表
    weight_now = weight[mask] #存储了sword和comid对应的权重关系表
    weight = weight_now[:,1:] #去除了前面存储的
    sword_final = data[:,0]
    nsword = len(sword_final)


    # --- 4. 映射COMID到索引 ---
    print('... creating index for each COMID ...')
    # 创建COMID到qq列索引的字典
    idnow_to_index = {cid: idx for idx, cid in enumerate(idnow)}
    # 将data中的COMID转换为qq中的列索引
    comid_indices = np.vectorize(lambda x: idnow_to_index.get(x, -1))(data[:, 1:])  # 无效索引为-1


    # --- ！！！5. 根据索引直接把qq值塞回数组，并利用矩阵的乘法来加速整个过程 ！！！ ----
    print('... extracting qq to matrix based on the index ...')
    
    valid_mask = comid_indices != -1  # 创建一个布尔掩码，标记有效索引
    rows, cols = np.where(valid_mask)  # 获取所有有效索引的行和列
    # 初始化结果数组
    result = np.full((nsword, 40, ntime), np.nan) #result = np.full((39528, 40, 1095), np.nan)

    result[rows, cols, :] = qq[:, comid_indices[valid_mask]].T
    result[result<0] = 0
    result[np.isnan(result)] = 0
    # --- 至此获得了维度为(nsword, 40, ntime)的三维矩阵，里面的值为根据索引取出来的qq ---



    # --- ！！！6. 利用NumPy 的广播和向量化操作，高效进行多维数组的运算，直接获得结果 ---
    print('... calculating weighted flow ...')
    weight_expanded = weight[:, :, np.newaxis]
    weighted_result = result * weight_expanded
    final_result = np.sum(weighted_result, axis=1)



    # --- 7. 创建输出NetCDF文件 ---
    fon = 'new_sword_runoff_mapped_pfaf_%02d.nc'%pfaf
    print('... writing to %s ...'%fon)
    # 创建输出文件
    output_nc = Dataset(fon, 'w', format='NETCDF4')
    output_nc.createDimension('sword_id', nsword)  # SWORD ID维度
    output_nc.createDimension('time', None)

    # SWORD ID变量
    sword_var = output_nc.createVariable('sword_id', 'i8', ('sword_id',))
    sword_var.long_name = "SWORD reach identifier"
    sword_var[:] = sword_final

    # 时间变量（假设2023年每日数据）
    start_date = datetime.datetime(2022,1,1)
    time_days = np.arange(ntime)
    time_var = output_nc.createVariable('time', 'f4', ('time',))
    time_var.units = f"days since {start_date.strftime('%Y-%m-%d %H:%M:%S')}"
    time_var.calendar = "proleptic_gregorian"
    time_var[:] = time_days

    runoff_var = output_nc.createVariable(
        'runoff', 
        'f4', 
        ('time', 'sword_id'), 
        zlib=True, 
        complevel=4, 
        fill_value=-9999.0
    )
    runoff_var.units = "m³/s"
    runoff_var.long_name = "Daily runoff mapped to SWORD reaches"
    runoff_var.missing_value = -9999.0


    # --- 4. 写入径流数据 ---
    runoff_var[:,:] = final_result.T #转置 

    output_nc.close()





