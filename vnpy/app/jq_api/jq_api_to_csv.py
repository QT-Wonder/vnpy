import jqdatasdk as jq
from datetime import datetime, timedelta
import time
import csv

start_counter = time.perf_counter()

jq.auth("13521365442", "QTWonder2020")


future_table = []
contracts = set()

underslying_symbol = 'IC'

start = datetime.strptime("2015-4-16", "%Y-%m-%d")
end = datetime.strptime("2020-12-31", "%Y-%m-%d")

start_contracts = jq.get_future_contracts(underslying_symbol, start.strftime("%Y-%m-%d"))
for contract in start_contracts:
    contracts.add(contract)

current = start 
while current <= end:
    current_contracts = jq.get_future_contracts(underslying_symbol, current.strftime("%Y-%m-%d"))
    if len(current_contracts)<1:
        break
    if current_contracts != start_contracts:
        for contract in current_contracts:
            contracts.add(contract)
        future_table.append(((start.strftime("%Y-%m-%d"), (current-timedelta(days=1)).strftime("%Y-%m-%d")), start_contracts))
        start = current
        start_contracts = current_contracts       
    current = current + timedelta(days=1)

contract_list = list(contracts)
contract_list.sort()

with open(underslying_symbol + "_contract_enddate.csv", mode='w', encoding='utf-8-sig', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')

    csv_writer.writerow(['Contract','EndDate'])
    for contract in contract_list:
        csv_writer.writerow([contract.replace('CCFX', 'CFFEX'), jq.get_security_info(contract).end_date])


with open(underslying_symbol + "_future_contract.csv", mode='w', encoding='utf-8-sig', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')

    csv_writer.writerow(['StartDate','EndDate', 'Future1', 'Future2', 'Future3', 'Future4'])
    for table in future_table:
        csv_writer.writerow([table[0][0], table[0][1], table[1][0].replace('CCFX', 'CFFEX') if len(table[1])>0 else '', table[1][1].replace('CCFX', 'CFFEX') if len(table[1])>1 else '', table[1][2].replace('CCFX', 'CFFEX') if len(table[1])>2 else '', table[1][3].replace('CCFX', 'CFFEX') if len(table[1])>3 else ''])


print(f'Done, take time: {time.perf_counter() - start_counter} seconds!')
