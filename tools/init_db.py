import sqlite3
import shutil
import pandas as pd

load_file='../travel_new.sqlite'
backup_file='../travel2.sqlite'
def update_dates():
    shutil.copyfile(backup_file, load_file) #拷贝文件

    conn = sqlite3.connect(load_file)

    values=pd.read_sql("select name from sqlite_master where type='table';", conn)['name'].values
    print(values)
    tdf={}
    for table_name in values:
        content=pd.read_sql(f"select * from {table_name}", conn)
        tdf[table_name]=content
    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    # 更新bookings表中的book_date
    tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff
    )

    # 需要更新的日期列
    datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    for column in datetime_columns:
        tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    # 将更新后的数据写回数据库
    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        del df  # 清理内存
    del tdf  # 清理内存

    conn.commit()
    conn.close()
