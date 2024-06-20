import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

st.title("Supply Chain Digital Twin")

excel_file_path="data/DigitalTwin.xlsx"
    
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx",'csv'])

def load_excel(file):
    return pd.ExcelFile(file)

def display_sheet(excel_file_path,sheet_name):
    try:
        df=pd.read_excel(excel_file_path,sheet_name=sheet_name)
        if df is not None and not df.empty:
            st.write(f"### {sheet_name}")
            st.dataframe(df)
            return df
        else:
            st.write("DataFrame is empty")
            return None
    except Exception as e:
         st.error(f"Error loading or displaying data: {str(e)}")
         return None


    






production_log_df = pd.DataFrame(columns=['Warehouse', 'Item', 'Required From Supplier'])
store_log_df = pd.DataFrame(columns=['Store', 'Item', 'Demand', 'Inv', 'Order fulfilled', 'Total from Warehouse', 'Total Trucks'])
removal_log_df = pd.DataFrame(columns=['Warehouse', 'Item','Inv','Quantity', 'Remaining Inventory'])



class Warehouse:
    def __init__(self, location, capacity,initial_inventory=None):
        self.location = location
        self.capacity = capacity
        self.inventory = initial_inventory if initial_inventory is not None else {}
        self.initial_inventory=self.inventory.copy()
        self.production_log = []

    def add_inventory(self, item, quantity):
        if item in self.inventory:
            self.inventory[item] += quantity
        else:
            self.inventory[item] = quantity

    def remove_inventory(self, item, quantity):
        global removal_log_df
        if item in self.inventory and self.inventory[item] >= quantity:
            self.inventory[item] -= quantity
            new_removal_record = pd.DataFrame({
                'Warehouse': [self.location],'Item': [item],'Inv':[self.initial_inventory[item]],'Quantity': [quantity],'Remaining Inventory': [self.inventory[item]]
            })
            if not new_removal_record.empty:
                removal_log_df = pd.concat([removal_log_df, new_removal_record], ignore_index=True)
        else:
            print("Error: Not enough inventory to fulfill order.")

    def simulate_production(self, item, quantity_needed):
        print(f"Simulating production for {quantity_needed} units of {item} in {self.location}")
        global production_log_df  # Reference the global DataFrame
        self.inventory[item] = self.inventory.get(item, 0) + quantity_needed
        # Create a new DataFrame for the new record
        new_record = pd.DataFrame({'Warehouse': [self.location], 'Item': [item],'Required From Supplier': [quantity_needed]})
        # Use pandas.concat to add the new record to the existing DataFrame
        if not new_record.empty:
            production_log_df = pd.concat([production_log_df, new_record], ignore_index=True)
        print(f"Produced {quantity_needed} units of {item} in {self.location}.")

    def print_inventory(self):
        print(f"Warehouse {self.location} current inventory:")
        for item, quantity in self.inventory.items():
            print(f"Item: {item}, Quantity: {quantity}")

class Truck:
    def __init__(self, location, capacity):
        self.location = location
        self.capacity = capacity
        self.cargo = {}

    def load_cargo(self, item, quantity):
        if sum(self.cargo.values()) + quantity <= self.capacity:
            self.cargo[item] = self.cargo.get(item, 0) + quantity
        else:
            print("Error: Not enough capacity to load cargo.")

    def unload_cargo(self, item, quantity):
        if item in self.cargo and self.cargo[item] >= quantity:
            self.cargo[item] -= quantity
        else:
            print("Error: Cargo not found.")

    def move_to(self, location):
        self.location = location

    def calculate_trucks_needed(total_space_required, truck_capacity):
        import math
        return math.ceil(total_space_required / truck_capacity)

class Store:
    def __init__(self, location, inventory, initial_inventory):
        self.location = location
        self.inventory = inventory.copy()  # Copy the inventory so changes do not affect the original dictionary
        self.initial_inventory = initial_inventory.copy()  # Keep a copy of the initial inventory levels
        self.warehouse = None

    def place_order(self, item, quantity,is_manual=True):
        if item not in self.inventory:
            print(f"Item {item} not found in store inventory.")
            return False
        global store_log_df
        if self.inventory[item] >= quantity:
            # Sufficient inventory to fulfill the order.
            self.inventory[item] -= quantity
            print(f"Order for {quantity} units of {item} fulfilled from store's inventory. Remaining: {self.inventory[item]}")
            rec=pd.DataFrame({'Store':[self.location],'Item':[item],'Demand':[quantity],'Inv':[self.initial_inventory[item]],'Order fulfilled':['yes'],'Total from Warehouse':[0], 'Total Trucks':[0]})
            if not rec.empty:
                store_log_df=pd.concat([store_log_df,rec],ignore_index=True)
            return True
        else:
            # Not enough inventory in the store, partial fulfillment and need to replenish from warehouse.
            print(f"Order partially fulfilled from the store: {self.inventory[item]}")
            additional_required = quantity - self.inventory[item]
            self.inventory[item] = 0
            print(f"Order for {additional_required} units of {item} to be fulfilled from Warehouse {self.warehouse.location}")

            # Replenish from the warehouse to fulfill the order and restore inventory to initial levels.
            replenishment_required = self.initial_inventory[item] - self.inventory[item] + additional_required
            if self.fulfill_order_from_warehouse(item, replenishment_required,is_manual):
               rec = pd.DataFrame({'Store': [self.location], 'Item': [item], 'Demand': [quantity], 'Inv': [self.initial_inventory[item]], 'Order fulfilled': ['no'], 'Total from Warehouse': [additional_required], 'Total Trucks': [0]})
               store_log_df = pd.concat([store_log_df, rec], ignore_index=True)
               if is_manual:
                    st.sidebar.success(f"Order for {quantity} units of {item} fulfilled from warehouse after partial fulfillment from store.")
                    return True  # Order fulfilled
            else:
                rec = pd.DataFrame({'Store': [self.location], 'Item': [item], 'Demand': [quantity], 'Inv': [self.initial_inventory[item]], 'Order fulfilled': ['no'], 'Total from Warehouse': [0], 'Total Trucks': [0]})
                store_log_df = pd.concat([store_log_df, rec], ignore_index=True)
                if is_manual:
                    st.sidebar.error(f"Order for {quantity} units of {item} could not be fulfilled completely.")
                    return False  

    def fulfill_order_from_warehouse(self, item, quantity, is_manual=True):
        truckloads_needed = 0
        print("Order partially fulfilled from store.", self.inventory)
        if item not in self.inventory:
            self.inventory[item] = 0
        remaining_quantity = max(0, quantity - self.inventory[item])
        global store_log_df
        # Calculate the total required including the amount to replenish
        reqd = quantity - remaining_quantity
        shortfall = reqd + remaining_quantity

        if self.warehouse:
            print(f"Total required from the warehouse (including replenishment): {shortfall}")

            if shortfall <= self.warehouse.inventory.get(item, 0):
                # Full fulfillment from warehouse
                self.warehouse.remove_inventory(item, shortfall)
                self.inventory[item] += remaining_quantity
                space_per_unit = item_capacities.get(item, 0)  # Get space required per unit of item
                total_space_required = shortfall * space_per_unit  # Total space required for the order

                # Calculate how many trucks are needed
                truckloads_needed = Truck.calculate_trucks_needed(total_space_required, truck.capacity)
                if is_manual:
                    st.sidebar.success(f"{truckloads_needed} trucks are required to transport {quantity} units of {item}.")
                    st.sidebar.success(f"{shortfall} units of {item} transferred from {self.warehouse.location} to store.")
                    return True
                rec=pd.DataFrame({'Store':[self.location],'Item':[item],'Demand':[quantity],'Inv':[self.initial_inventory[item]],'Order fulfilled':['no'],'Total from Warehouse':[shortfall], 'Total Trucks':[truckloads_needed]})
                store_log_df=pd.concat([store_log_df,rec],ignore_index=True)
                
            else:
                # Partial fulfillment from warehouse
                additional_required = shortfall - self.warehouse.inventory.get(item, 0)
                print(f"Not enough stock in {self.warehouse.location} warehouse. Additional required: {additional_required}")
                if additional_required > 0:
                    self.warehouse.simulate_production(item, additional_required)

                # Transfer whatever is available in the warehouse to fulfill the order
                transfer_quantity = min(shortfall, self.warehouse.inventory.get(item, 0))
                self.warehouse.remove_inventory(item, transfer_quantity)
                self.inventory[item] += transfer_quantity
                if is_manual:
                    st.sidebar.success(f"After production, {transfer_quantity} units of {item} transferred from {self.warehouse.location} to store.")
                    return True
                space_per_unit = item_capacities.get(item, 0)  # Get space required per unit of item
                total_space_required = transfer_quantity * space_per_unit  # Total space required for the order

                # Calculate how many trucks are needed
                truckloads_needed = Truck.calculate_trucks_needed(total_space_required, truck.capacity)
                if is_manual:
                    st.sidebar.success(f"{truckloads_needed} trucks are required to transport {quantity} units of {item}.")
                    st.sidebar.success(f"{transfer_quantity} units of {item} transferred from {self.warehouse.location} to store.")
                    return True
                rec=pd.DataFrame({'Store':[self.location],'Item':[item],'Demand':[quantity],'Inv':[self.initial_inventory[item]],'Order fulfilled':['no'],'Total from Warehouse':[transfer_quantity], 'Total Trucks':[truckloads_needed]})
                if not rec.empty:store_log_df=pd.concat([store_log_df,rec],ignore_index=True)
                
        else:
            st.sidebar.error(f"No warehouse linked to {self.location}.")
            return False




truck = Truck("New York", 250)



def cost_calc(excel_file_path,sheet_name):
  d1=pd.read_excel(excel_file_path,sheet_name=sheet_name)
  return d1

c_cal=cost_calc(excel_file_path,'Cost')
print(c_cal)


def load_item_capacities(excel_file_path, sheet_name):
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    capacities = df.set_index('Item')['Capacity'].to_dict()
    return capacities

item_capacities = load_item_capacities(excel_file_path,'item_capacity')

def load_warehouse_capacity(excel_file_path, sheet_name):
    capacity_df = pd.read_excel(excel_file_path, sheet_name)
    capacity_dict = capacity_df.set_index('location')['capacity'].to_dict()
    return capacity_dict


# Load warehouse capacity from the Excel sheet
warehouse_capacity = load_warehouse_capacity(excel_file_path, 'warehouse')
# Verify the data
print(warehouse_capacity)

def load_warehouse_demand(excel_file_path, sheet_name):
    # Read the warehouse demand sheet
    demand_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)


    demand_dict = {}
    for index, row in demand_df.iterrows():
        warehouse = row['Warehouse']
        item = row['Item']
        quantity = row['Quantity']

        if warehouse not in demand_dict:
            demand_dict[warehouse] = {}

        demand_dict[warehouse][item] = quantity

    return demand_dict

warehouse_demand = load_warehouse_demand(excel_file_path, 'warehouse_inventory')
print(warehouse_demand)

warehouses = {}
for warehouse_name, cap in warehouse_capacity.items():
    initial_inventory = warehouse_demand.get(warehouse_name, {})
    warehouses[warehouse_name] = Warehouse(warehouse_name, cap, initial_inventory)


def load_initial_store_inventory(excel_file_path,sheet_name):
    initial_inventory_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    initial_inventory_dict = {}
    for index, row in initial_inventory_df.iterrows():
        store = row['Store']
        item = row['Item']
        quantity = row['Quantity']
        if store not in initial_inventory_dict:
            initial_inventory_dict[store] = {}
        initial_inventory_dict[store][item] = quantity
    return initial_inventory_dict

initial_store_inventory = load_initial_store_inventory(excel_file_path,'Store_inventory')


print(initial_store_inventory)

stores = {}



for store_id, inventory in initial_store_inventory.items():
    stores[store_id] = Store(store_id, inventory, inventory)
    initial_inv = initial_store_inventory.get(store_id, {})
    stores[store_id] = Store(store_id, inventory, initial_inv)

    if store_id in [1, 2, 3]:  # Link stores 1, 2, and 3 to 'Warehouse 1'
        linked_warehouse = warehouses.get('Warehouse 1')
    elif store_id in [4, 5, 6]:  # Link stores 4, 5, and 6 to 'Warehouse 2'
        linked_warehouse = warehouses.get('Warehouse 2')
    else:
        linked_warehouse = None  # Fallback, in case there are more stores not accounted for



    stores[store_id].warehouse = linked_warehouse


# Function to link stores to warehouses based on the store ID
def link_stores_to_warehouses():
    for store_id in stores:
        if store_id in [1, 2, 3]:
            stores[store_id].warehouse = warehouses.get('Warehouse 1')
        elif store_id in [4, 5, 6]:
            stores[store_id].warehouse = warehouses.get('Warehouse 2')


link_stores_to_warehouses()

def load_warehouse_fluctuations(excel_file_path, sheet_name):
    fluctuation_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    fluctuation_dict = fluctuation_df.set_index(['Warehouse', 'Item'])['Fluctuation'].to_dict()
    return fluctuation_dict

warehouse_fluctuations = load_warehouse_fluctuations(excel_file_path, 'warehouse_inventory')

def apply_fluctuations_to_inventory(warehouses, fluctuation_dict):
    for warehouse_name, warehouse in warehouses.items():
        for item, initial_quantity in warehouse.initial_inventory.items():
            fluc = fluctuation_dict.get((warehouse_name, item), 0)
            min_qty = initial_quantity
            max_qty = np.rint(initial_quantity + fluc )
            random_qty = random.randint(min_qty, max_qty)
            warehouse.initial_inventory[item] = random_qty
            warehouse.inventory[item] = random_qty  # Updating current inventory to fluctuated value

apply_fluctuations_to_inventory(warehouses, warehouse_fluctuations)


def update_warehouse_inventory(excel_file_path, sheet_name):
    # Reset inventory to the initial state before the iteration
    for warehouse_name in warehouses:
        warehouses[warehouse_name].inventory = warehouses[warehouse_name].initial_inventory.copy()




def manual_order_entry(initial_store_inventory):
    st.sidebar.header("Manual Order Entry")

    store_id = st.sidebar.number_input("Enter Store ID (1-6)", min_value=1, max_value=6)
    item = st.sidebar.selectbox("Enter the item name", ["A", "B", "C"])
    quantity = st.sidebar.number_input("Enter the quantity", min_value=1)

    if st.sidebar.button('Place Order'):
        selected_store = stores.get(store_id)
        if not selected_store:
            st.sidebar.error(f"Invalid Store ID. Please enter a valid Store ID.")
            return
        
        if item not in initial_store_inventory.get(store_id, {}):
            st.sidebar.error(f"Item {item} does not exist in Store {store_id}'s inventory. Please try again.")
            return

        if selected_store.place_order(item, quantity,is_manual=True):
            st.sidebar.success(f"Store {store_id} ordered {quantity} units of item {item}.")
        else:
             st.sidebar.warning(f"Store {store_id} ordered {quantity} units of item {item}, but not enough inventory in store. Additional items ordered from warehouse to fulfill the demand.")
        selected_store.warehouse.print_inventory()
        st.sidebar.write(selected_store.warehouse.print_inventory())
        #st.sidebar.error(f"Not enough {item} in the warehouse to fulfill the order.")

summary_columns = [
    'Iteration', 'Total Carrying Cost Stores 1-3', 'Total Shipping Cost Stores 1-3',
    'Total Carrying Cost Stores 4-6', 'Total Shipping Cost Stores 4-6', 'Total Carrying Cost All Stores',
    'Total Shipping Cost All Stores', 'Total Carrying Cost Warehouses', 'Total Shipping Cost Warehouses',
    'Store Fulfillment %', 'Warehouse Fulfillment %', 'Warehouse 1 Capacity Utilised %', 'Warehouse 2 Capacity Utilised %'
]
summary_df = pd.DataFrame(columns=summary_columns)

def reset_warehouse_inventory(warehouses):
    for warehouse in warehouses.values():
        warehouse.inventory = warehouse.initial_inventory.copy()

def reset_store_inventory(stores):
    for store in stores.values():
        store.inventory = store.initial_inventory.copy()

#  simulate_excel_input_orders function
class OrderSimulator:
    def __init__(self, excel_file_path, quantity_multiplier, num_iterations):
        self.excel_file_path = excel_file_path
        self.quantity_multiplier = quantity_multiplier
        self.num_iterations = num_iterations
        self.orders_df = pd.read_excel(excel_file_path)
        self.summary_df=pd.DataFrame()
        self.initialize_summary_df()
        self.summary_data=[]

    def initialize_summary_df(self):
        if 'summary_df' not in st.session_state:
            numeric_columns = ['Iteration',
                               'Total Carrying Cost Stores 1-3',
                               'Total Shipping Cost Stores 1-3',
                               'Total Carrying Cost Stores 4-6',
                               'Total Shipping Cost Stores 4-6',
                               'Total Carrying Cost All Stores',
                               'Total Shipping Cost All Stores',
                               'Total Carrying Cost Warehouses',
                               'Total Shipping Cost Warehouses',
                               'Store Fulfillment %',
                               'Warehouse Fulfillment %',
                               'Warehouse 1 Capacity Utilised %',
                               'Warehouse 2 Capacity Utilised %']
            st.session_state['summary_df'] = pd.DataFrame(columns=numeric_columns).astype(float)
        self.summary_df = st.session_state['summary_df']

    def simulate_excel_input_orders(self):
        orders_df = pd.read_excel(excel_file_path)
        



        for iteration in range(self.num_iterations):
            global store_log_df,removal_log_df,production_log_df
            print(f"\nIteration {iteration + 1}\n")
            store_log_df = pd.DataFrame(columns=['Store', 'Item', 'Demand', 'Inv', 'Order fulfilled', 'Total from Warehouse', 'Total Trucks'])
            removal_log_df = pd.DataFrame(columns=['Warehouse', 'Item','Inv','Quantity', 'Remaining Inventory'])
            production_log_df = pd.DataFrame(columns=['Warehouse', 'Item', 'Required From Supplier'])

            reset_warehouse_inventory(warehouses)
            reset_store_inventory(stores)

            apply_fluctuations_to_inventory(warehouses, warehouse_fluctuations)
            


            for index, row in self.orders_df.iterrows():
                item = row['Item'].upper()
                store_selection = row['Store']
                base_quantity = row['Quantity']
                fluc=row['Fluctuation']

                min_1=(base_quantity - base_quantity*(fluc/100))
                max_1=(base_quantity + base_quantity* (fluc/100))

                min_quantity = np.rint(min_1)
                max_quantity = np.rint(max_1)
                print("min",min_quantity)
                print("max",max_quantity)

                min_quantity = max(min_quantity, 0)

                quantity = random.randrange(min_quantity, max_quantity) * quantity_multiplier


                selected_store = stores.get(store_selection)

                if selected_store:
                    selected_store.place_order(item, quantity,is_manual=False)
                    print(f"Store {store_selection} ({selected_store.location}) ordered {quantity} units of item {item}.")
                    print(f"iteration number:{index+1} completed\n")
                else:
                    print(f"Invalid store selection: {store_selection}")
                    print(f"iteration number:{index+1} completed\n")

            carrying_cost_dict = c_cal.set_index("Item")["Inventory Carrying cost"].to_dict()
            shipping_cost_dict = c_cal.set_index("Item")["Shipping cost"].to_dict()


                # Group 'merged_df' by 'Warehouse' and sum the 'Quantity' column
            # Apply carrying cost to store inventory
            store_log_df["Carrying Cost"] = store_log_df["Item"].map(carrying_cost_dict) * store_log_df["Inv"]
            carry_store=store_log_df["Carrying Cost"].sum()

                    # Apply shipping cost to the store inventory
            store_log_df["Shipping Cost"] = store_log_df["Item"].map(shipping_cost_dict) * store_log_df["Total from Warehouse"]
            ship_store=store_log_df["Shipping Cost"].sum()
            print("Store Log:\n")
            print(store_log_df)

            g1=store_log_df[store_log_df['Store'].isin([1, 2, 3])]
            g2=store_log_df[store_log_df['Store'].isin([4, 5, 6])]

            g1_cc=g1['Carrying Cost'].sum()
            g1_sc=g1['Shipping Cost'].sum()

            print(f"\n The Total Carrying Cost For Stores-1,2,3 : {g1_cc}")
            print(f"\n The Total Shipping Cost For Stores-1,2,3 : {g1_sc}")

            g2_cc=g2['Carrying Cost'].sum()
            g2_sc=g2['Shipping Cost'].sum()

            print(f"\n The Total Carrying Cost For Stores-4,5,6 : {g2_cc}")
            print(f"\n The Total Carrying Cost For Stores-4,5,6 : {g2_sc}")

            print(f"\nThe Total Carrying Cost For All The Stores : {carry_store}")
            print(f"\nThe Total Shipping Cost For All The Stores : {ship_store}")

            # Sort and group by 'Warehouse' and 'Item'
            sorted_df = production_log_df.sort_values(by='Warehouse')
            summed_df = sorted_df.groupby(['Warehouse', 'Item'])['Required From Supplier'].sum().reset_index()

            # Group and calculate fulfilled status
            r_df = removal_log_df.groupby(['Warehouse', 'Item', 'Inv'])['Quantity'].sum().reset_index()
            r_df['Fulfilled'] = r_df.apply(lambda row: 'yes' if row['Quantity'] < row['Inv'] else 'no', axis=1)

            # Merge dataframes on 'Warehouse' and 'Item'
            merged_df = pd.merge(r_df, summed_df, on=['Warehouse', 'Item'], how='left')

            # Set 'Required From Supplier' to 0 if 'Fulfilled' is 'yes'
            merged_df['Required From Supplier'] = merged_df.apply(lambda row: 0 if row['Fulfilled'] == 'yes' else row['Required From Supplier'],
                        axis=1)

            warehouse_totals = merged_df.groupby('Warehouse')['Quantity'].sum().reset_index()

                    # Apply carrying cost to warehouse demand
            merged_df["Carrying Cost"] = merged_df["Item"].map(carrying_cost_dict) * merged_df["Inv"]
            c_wh= merged_df["Carrying Cost"].sum()

            merged_df["Shipping Cost"] = merged_df["Item"].map(shipping_cost_dict) * merged_df["Required From Supplier"]
            s_wh=merged_df["Shipping Cost"].sum()

                    # Display the merged dataframe
            print("\nWarehouse Log:\n")
            print(merged_df)

            print(f"\nThe Total Carrying Cost For The Warehouses : {c_wh}")
            print(f"\nThe Total Shipping Cost For The Warehouses : {s_wh}")

                    # Calculate percentage of 'yes' (fulfilled demand)
            total_fulfilled_store = store_log_df['Order fulfilled'].value_counts().get('yes', 0)
            total_orders_store = len(store_log_df)

            total_fulfilled_warehouse = merged_df['Fulfilled'].value_counts().get('yes', 0)
            total_entries_warehouse = len(merged_df)

                    # Calculate the percentage of fulfilled demand
            store_fulfillment_percentage = (total_fulfilled_store / total_orders_store) * 100
            if total_entries_warehouse>0:
                warehouse_fulfillment_percentage = (total_fulfilled_warehouse / total_entries_warehouse) * 100
            else:
                warehouse_fulfillment_percentage= 0
            

                    # Print fulfillment percentage
            print(f"\nStore Fulfillment : {store_fulfillment_percentage:.2f}%")
            print(f"Warehouse Fulfillment : {warehouse_fulfillment_percentage:.2f}%")
            warehouse_utilization = {}


                    # Loop through the warehouse capacities and compare with the total quantities
            for index, row in warehouse_totals.iterrows():
                warehouse_location = row['Warehouse']
                total_quantity = row['Quantity']
                capacity = warehouse_capacity.get(warehouse_location, 0)
            if total_quantity <= capacity:
                print(f"{warehouse_location} can fulfill demand. Total quantity: {total_quantity}, Capacity: {capacity}.")
                percent_fulfill=(total_quantity/capacity)*100
                print(f"Total {warehouse_location} Capacity Utilised:{percent_fulfill:.2f}%")
            else:
                print(f"{warehouse_location} cannot fulfill demand. Total quantity: {total_quantity}, Capacity: {capacity}.")
                percent_fulfill = 0

            warehouse_utilization[f"{warehouse_location} Utilization %"] = percent_fulfill

            iteration_data = {
                'Iteration': iteration + 1,
                'Total Carrying Cost Stores 1-3': g1_cc,
                'Total Shipping Cost Stores 1-3': g1_sc,
                'Total Carrying Cost Stores 4-6': g2_cc,
                'Total Shipping Cost Stores 4-6': g2_sc,
                'Total Carrying Cost All Stores': carry_store,
                'Total Shipping Cost All Stores': ship_store,
                'Total Carrying Cost Warehouses': c_wh,
                'Total Shipping Cost Warehouses': s_wh,
                'Store Fulfillment %': store_fulfillment_percentage,
                'Warehouse Fulfillment %': warehouse_fulfillment_percentage,
                'Warehouse 1 Capacity Utilised %': warehouse_utilization.get('Warehouse 1 Utilization %', 0),
                'Warehouse 2 Capacity Utilised %': warehouse_utilization.get('Warehouse 2 Utilization %', 0)
            }
            
            self.summary_data.append(iteration_data)
            self.summary_df = pd.DataFrame(self.summary_data)
        self.summary_df = self.summary_df.astype(float)
        print("\nSummary DataFrame:\n")
        print(self.summary_df)
        st.session_state['summary_df'] = self.summary_df
        return self.summary_df

        

        
    def display_summary_stats(self):       

                max_df = pd.DataFrame([self.summary_df[['Total Carrying Cost Stores 1-3', 
                                            'Total Shipping Cost Stores 1-3', 
                                            'Total Carrying Cost Stores 4-6', 
                                            'Total Shipping Cost Stores 4-6', 
                                            'Total Carrying Cost All Stores', 
                                            'Total Shipping Cost All Stores', 
                                            'Total Carrying Cost Warehouses', 
                                            'Total Shipping Cost Warehouses', 
                                            'Store Fulfillment %', 
                                            'Warehouse Fulfillment %', 
                                            'Warehouse 1 Capacity Utilised %', 
                                            'Warehouse 2 Capacity Utilised %']].max()], 
                                index=["Max"])

            # Create a DataFrame for minimum values
                min_df = pd.DataFrame([self.summary_df[['Total Carrying Cost Stores 1-3', 
                                                'Total Shipping Cost Stores 1-3', 
                                                'Total Carrying Cost Stores 4-6', 
                                                'Total Shipping Cost Stores 4-6', 
                                                'Total Carrying Cost All Stores', 
                                                'Total Shipping Cost All Stores', 
                                                'Total Carrying Cost Warehouses', 
                                                'Total Shipping Cost Warehouses', 
                                                'Store Fulfillment %', 
                                                'Warehouse Fulfillment %', 
                                                'Warehouse 1 Capacity Utilised %', 
                                                'Warehouse 2 Capacity Utilised %']].min()], 
                                    index=["Min"])

            # Create a DataFrame for average values
                avg_df = pd.DataFrame([self.summary_df[['Total Carrying Cost Stores 1-3', 
                                                'Total Shipping Cost Stores 1-3', 
                                                'Total Carrying Cost Stores 4-6', 
                                                'Total Shipping Cost Stores 4-6', 
                                                'Total Carrying Cost All Stores', 
                                                'Total Shipping Cost All Stores', 
                                                'Total Carrying Cost Warehouses', 
                                                'Total Shipping Cost Warehouses', 
                                                'Store Fulfillment %', 
                                                'Warehouse Fulfillment %', 
                                                'Warehouse 1 Capacity Utilised %', 
                                                'Warehouse 2 Capacity Utilised %']].mean()], 
                                    index=["Average"])
                
                summary_stats_df = pd.concat([max_df, min_df, avg_df])
                print(summary_stats_df)
                return summary_stats_df
    
        

    
    def plot_summary_data(self, x_axis, y_axis, plot_type):
        try:
            fig, ax = plt.subplots()
            if plot_type == "Line":
                self.summary_df.plot(x=x_axis, y=y_axis, ax=ax, kind='line')
            elif plot_type == "Bar":
                self.summary_df.plot(x=x_axis, y=y_axis, ax=ax, kind='bar')
            elif plot_type == "Scatter":
                self.summary_df.plot(x=x_axis, y=y_axis, ax=ax, kind='scatter')
            elif plot_type == "Histogram":
                self.summary_df[y_axis].plot(kind='hist', ax=ax)
            elif plot_type == "Boxplot":
                sns.boxplot(x=x_axis, y=y_axis, data=self.summary_df, ax=ax)
            elif plot_type == "Heatmap":
                sns.heatmap(self.summary_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            elif plot_type == "Area":
                self.summary_df.plot(x=x_axis, y=y_axis, ax=ax, kind='area')
            elif plot_type == "Pie":
                self.summary_df[y_axis].plot(kind='pie', autopct='%1.1f%%', ax=ax)
            elif plot_type == "Violin":
                sns.violinplot(x=x_axis, y=y_axis, data=self.summary_df, ax=ax)

            st.pyplot(fig)  # Display the plot in Streamlit
            plt.close(fig)  # Close the figure to free up memory

        except Exception as e:
            st.error(f"Error generating plot: {e}")


        

   



if __name__ == "__main__":
    if 'file_to_load' not in st.session_state:
        st.session_state['file_to_load'] = excel_file_path
    if 'sheet_names' not in st.session_state:
        st.session_state['sheet_names'] = []
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

    
    if st.button("Run"):
        if uploaded_file is not None:
            file_to_load= uploaded_file

        excel_file_path = load_excel(st.session_state['file_to_load'])
        st.session_state['sheet_names'] = excel_file_path.sheet_names

    df = st.session_state.get('df', pd.DataFrame())   
    with st.expander("Select Sheet and Display DataFrame", expanded=True):
            m1 = st.selectbox("Select Sheet", st.session_state['sheet_names'])
            st.session_state['df'] = display_sheet(st.session_state['file_to_load'], m1)
            df = st.session_state['df']
    
    #st.header("Visualizations")
    with st.expander("Plot Options", expanded=True):
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)
        plot_type = st.selectbox("Select Plot Type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Heatmap", "Violin"])

    if st.button("Generate Plot"):
            try:
                fig, ax = plt.subplots()
                if plot_type == "Line":
                    df.plot(x=x_axis, y=y_axis, ax=ax, kind='line')
                elif plot_type == "Bar":
                    df.plot(x=x_axis, y=y_axis, ax=ax, kind='bar')
                elif plot_type == "Scatter":
                    df.plot(x=x_axis, y=y_axis, ax=ax, kind='scatter')
                elif plot_type == "Histogram":
                    df[y_axis].plot(kind='hist', ax=ax)
                elif plot_type == "Boxplot":
                    df.boxplot(column=y_axis, ax=ax)
                elif plot_type == "Heatmap":
                    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                elif plot_type == "Violin":
                    sns.violinplot(x=x_axis, y=y_axis, data=df, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating plot: {e}")
    else:
            st.write("Press the Run button to load and view data")

    
    st. header("Simulation Controls")
    mode = st.selectbox("Select Mode", ["Automatic Order Entry","Manual Order Entry"])
    if mode == "Manual Order Entry":
            manual_order_entry(initial_store_inventory)
    elif mode == "Automatic Order Entry":
            quantity_multiplier = st.sidebar.number_input("Quantity Multiplier", min_value=1, value=1)
            num_iterations = st.sidebar.slider("Number of Iterations", min_value=1, value=1)
            if st.sidebar.button("Run Simulation"):
                try:
                        simulator = OrderSimulator(excel_file_path, quantity_multiplier, num_iterations)
                        summary_df = simulator.simulate_excel_input_orders()

                        st.subheader("Simulation Summary")
                        st.dataframe(summary_df)
                        st.session_state['summary_df'] = summary_df
                        st.write(simulator.display_summary_stats())
                except Exception as e:
                    st.error(f"Error running simulation: {e}")

                        

            if 'summary_df' in st.session_state:
                    summary_df = st.session_state['summary_df']
                    st.sidebar.header("Plot Options")
                    x_axis_sim = st.sidebar.selectbox("Select X-axis", summary_df.columns)
                    y_axis_sim = st.sidebar.selectbox("Select Y-axis", summary_df.columns)
                    plot_type_sim = st.sidebar.selectbox("Select Plot Type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Heatmap", "Area", "Pie", "Violin"])

            if st.sidebar.button("Plot"):
                simulator = OrderSimulator(excel_file_path, quantity_multiplier, num_iterations)
                simulator.summary_df = summary_df
                simulator.plot_summary_data(x_axis_sim, y_axis_sim, plot_type_sim)

                
    
    #st.write("Summary DataFrame")
    #st.write(st.session_state['summary_df'])
    print(summary_df.dtypes)
    #summary_df[x_axis] = pd.to_numeric(summary_df[x_axis], errors='coerce')
    #summary_df[y_axis] = pd.to_numeric(summary_df[y_axis], errors='coerce')

    
    
    
