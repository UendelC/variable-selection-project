import csv
import random

header = [ 'sessionNo', 'startHour', 'startWeekday', 'duration', 'cCount', 'cMinPrice', 'cMaxPrice', 'cSumPrice', 'bCount', 'bMinPrice', 'bMaxPrice', 'bSumPrice', 'bStep', 'onlineStatus', 'availability', 'customerId', 'maxVal', 'customerScore', 'accountLifetime', 'payments', 'age', 'address', 'lastOrder', 'order' ]

# Ações do usuário que encerram uma transação: clicar em um produto, adicionar produto à cesta, remover produto da cesta
# Colunas diretamente afetadas por essas ações: cCount, bCount

def sumCalc( minPrice, maxPrice, productCount):

    if type(minPrice) == str:
        minPrice = 0

    if type(maxPrice) == str:
        maxPrice = 150

    if minPrice == maxPrice:
                sumPrice = minPrice*productCount

    else:
        sumPrice = maxPrice + minPrice

        for product in range( 0, productCount ):
            sumPrice += random.uniform(minPrice, maxPrice)

    sumPrice = random.choice( [ sumPrice, '?' ] )
    sumPrice = format( sumPrice, '.2f' ) if type( sumPrice ) == float else sumPrice

    return sumPrice

def randomWithMissing( minValue, maxValue ):

    if type(minValue) == str:
        minValue = 0
    if type(maxValue) == str:
        maxValue = 150

    value = random.choice( [random.uniform(minValue, maxValue), '?'] )
    value = format( value, '.2f' ) if type( value ) == float else value

    return value


with open('data/transact_train.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    for sessionNo in range(0, 10):
        # Define o início da sessão
        startHour       = random.randint(0, 23)
        startWeekday    = random.randint(1, 7)
        duration        = 0
        customerId      = random.choice( [ '?', ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) ] )
        
        if customerId == '?':
            maxVal = customerScore = accountLifetime = payments = age = address = lastOrder = '?'
        else:
            maxVal          = random.choice( [ '?', random.randint(1, 30000) ] )
            customerScore   = random.choice( [ '?', random.randint(0, 100) ] )
            accountLifetime = random.choice( [ '?', random.randint(0, 120) ] )
            payments        = random.choice( [ '?', random.randint(0, 120) ] )
            age             = random.choice( [ '?', random.randint(16, 89) ] )
            address         = random.choice( [ '?', random.randint(1, 3) ] )
            lastOrder       = random.choice( [ '?', random.randint( 0, accountLifetime*30 ) ] )
        
        order           = random.choice( [ 'y', 'n' ] )

        transactions = random.randint(1, 10)

        # Define as transações que ocorrem na sessão
        for transaction in range(0, transactions):

            duration    += random.uniform(0, 1800)
            cCount       = random.randint(0, 10)
            cMinPrice    = 0 if cCount == 0 else randomWithMissing( 10, 150 )
            cMaxPrice    = 0 if cCount == 0 else randomWithMissing( cMinPrice, 150 )
            cSumPrice    = 0 if cCount == 0 else sumCalc( cMinPrice, cMaxPrice, cCount)
            bCount       = random.randint(0, 10)
            bMinPrice    = 0 if bCount == 0 else randomWithMissing( 0, 150 )
            bMaxPrice    = 0 if bCount == 0 else randomWithMissing( bMinPrice, 150 )
            bSumPrice    = 0 if bCount == 0 else sumCalc( bMinPrice, bMaxPrice, bCount )
            bStep        = random.choice( [ 1, 2, 3, 4, 5, '?' ] )
            onlineStatus = random.choice( [ 'y', 'n', '?' ] )
            availability = random.choice( [ 'completely orderable', 'completely not orderable', 'mainly orderable', '?' ] ) 
            

            data = [ 
                sessionNo, 
                startHour, 
                startWeekday, 
                format( duration, ".2f" ), 
                cCount, 
                cMinPrice, 
                cMaxPrice, 
                cSumPrice, 
                bCount, 
                bMinPrice, 
                bMaxPrice, 
                bSumPrice,
                bStep,
                onlineStatus,
                availability,
                customerId,
                maxVal,
                customerScore,
                accountLifetime,
                payments,
                age,
                address,
                lastOrder,
                order
                ]

            writer.writerow(data)