import geopandas as gpd
import pandas as pd
from shapely.geometry import box

def get_coastline_vectors(shpfile_path):
    gdf=gpd.read_file(shpfile_path) 
    gdf=gdf.to_crs(epsg=4326) #coodinate system used

    irish_sea_bbox=box(-8.0, 50, -2.2, 56.5) #coodinates surroudning irish sea image.
    gdf_clipped=gdf.clip(irish_sea_bbox)

    def extract_vertices(geom):
        if geom is None:
            return []
        if geom.geom_type=="LineString":
            return list(geom.coords)
        elif geom.geom_type=="MultiLineString":
            points=[]
            for line in geom.geoms: #get points for each line string
                for point in line.coords:
                    points.append(point)
            return points
        else:
            return []

    all_points=[]
    for geom in gdf_clipped.geometry:
        for points in extract_vertices(geom):
            all_points.append(points)
    all_points.append([54.4472,-5.5927]) #A point in northern ireland which needs to be removed but is not included in the mask.
    return pd.DataFrame(all_points, columns=["longitude", "latitude"])


