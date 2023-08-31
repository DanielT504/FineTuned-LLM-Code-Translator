def Loss ( SP , P ) : NEW_LINE INDENT loss = 0 NEW_LINE loss = ( ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ) NEW_LINE print ( " Loss ▁ = " , round ( loss , 3 ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT SP , P = 2400 , 30 NEW_LINE DEDENT
Loss ( SP , P ) NEW_LINE
MAXN = 1000001 NEW_LINE
spf = [ i for i in range ( MAXN ) ] NEW_LINE
hash1 = [ 0 for i in range ( MAXN ) ] NEW_LINE
def sieve ( ) : NEW_LINE
for i in range ( 4 , MAXN , 2 ) : NEW_LINE INDENT spf [ i ] = 2 NEW_LINE DEDENT
for i in range ( 3 , MAXN ) : NEW_LINE INDENT if i * i >= MAXN : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
if ( spf [ i ] == i ) : NEW_LINE INDENT for j in range ( i * i , MAXN , i ) : NEW_LINE DEDENT
if ( spf [ j ] == j ) : NEW_LINE INDENT spf [ j ] = i NEW_LINE DEDENT
def getFactorization ( x ) : NEW_LINE INDENT while ( x != 1 ) : NEW_LINE INDENT temp = spf [ x ] NEW_LINE if ( x % temp == 0 ) : NEW_LINE DEDENT DEDENT
hash1 [ spf [ x ] ] += 1 NEW_LINE x = x // spf [ x ] NEW_LINE while ( x % temp == 0 ) : NEW_LINE x = x // temp NEW_LINE
def check ( x ) : NEW_LINE INDENT while ( x != 1 ) : NEW_LINE INDENT temp = spf [ x ] NEW_LINE DEDENT DEDENT
if ( x % temp == 0 and hash1 [ temp ] > 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT while ( x % temp == 0 ) : NEW_LINE INDENT x = x // temp NEW_LINE DEDENT return True NEW_LINE
def hasValidNum ( arr , n ) : NEW_LINE
sieve ( ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT getFactorization ( arr [ i ] ) NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( check ( arr [ i ] ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT DEDENT return False NEW_LINE
arr = [ 2 , 8 , 4 , 10 , 6 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE if ( hasValidNum ( arr , n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def countWays ( N ) : NEW_LINE
E = ( N * ( N - 1 ) ) / 2 NEW_LINE if ( N == 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return int ( pow ( 2 , E - 1 ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 4 NEW_LINE print ( countWays ( N ) ) NEW_LINE DEDENT
def minAbsDiff ( n ) : NEW_LINE INDENT mod = n % 4 ; NEW_LINE if ( mod == 0 or mod == 3 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT return 1 ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; NEW_LINE print ( minAbsDiff ( n ) ) NEW_LINE DEDENT
def check ( s ) : NEW_LINE
freq = [ 0 ] * 10 NEW_LINE while ( s != 0 ) : NEW_LINE
r = s % 10 NEW_LINE
s = s // 10 NEW_LINE
freq [ r ] += 1 NEW_LINE xor = 0 NEW_LINE
for i in range ( 10 ) : NEW_LINE INDENT xor = xor ^ freq [ i ] NEW_LINE DEDENT if ( xor == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
s = 122233 NEW_LINE if ( check ( s ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def printLines ( n , k ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT print ( k * ( 6 * i + 1 ) , k * ( 6 * i + 2 ) , k * ( 6 * i + 3 ) , k * ( 6 * i + 5 ) ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n , k = 2 , 2 NEW_LINE printLines ( n , k ) NEW_LINE DEDENT
def calculateSum ( n ) : NEW_LINE
return ( 2 ** ( n + 1 ) + n - 2 ) NEW_LINE
n = 4 NEW_LINE
print ( " Sum ▁ = " , calculateSum ( n ) ) NEW_LINE
def partitions ( n ) : NEW_LINE INDENT p = [ 0 ] * ( n + 1 ) NEW_LINE DEDENT
p [ 0 ] = 1 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT k = 1 NEW_LINE while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) : NEW_LINE INDENT p [ i ] += ( ( 1 if k % 2 else - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) // 2 ] ) NEW_LINE if ( k > 0 ) : NEW_LINE INDENT k *= - 1 NEW_LINE DEDENT else : NEW_LINE INDENT k = 1 - k NEW_LINE DEDENT DEDENT DEDENT return p [ n ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 20 NEW_LINE print ( partitions ( N ) ) NEW_LINE DEDENT
def countPaths ( n , m ) : NEW_LINE
if ( n == 0 or m == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) NEW_LINE
n = 3 NEW_LINE m = 2 NEW_LINE print ( " ▁ Number ▁ of ▁ Paths ▁ " , countPaths ( n , m ) ) NEW_LINE
def getMaxGold ( gold , m , n ) : NEW_LINE
goldTable = [ [ 0 for i in range ( n ) ] for j in range ( m ) ] NEW_LINE for col in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT for row in range ( m ) : NEW_LINE DEDENT
if ( col == n - 1 ) : NEW_LINE INDENT right = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right = goldTable [ row ] [ col + 1 ] NEW_LINE DEDENT
if ( row == 0 or col == n - 1 ) : NEW_LINE INDENT right_up = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right_up = goldTable [ row - 1 ] [ col + 1 ] NEW_LINE DEDENT
if ( row == m - 1 or col == n - 1 ) : NEW_LINE INDENT right_down = 0 NEW_LINE DEDENT else : NEW_LINE INDENT right_down = goldTable [ row + 1 ] [ col + 1 ] NEW_LINE DEDENT
goldTable [ row ] [ col ] = gold [ row ] [ col ] + max ( right , right_up , right_down ) NEW_LINE
res = goldTable [ 0 ] [ 0 ] NEW_LINE for i in range ( 1 , m ) : NEW_LINE INDENT res = max ( res , goldTable [ i ] [ 0 ] ) NEW_LINE DEDENT return res NEW_LINE
gold = [ [ 1 , 3 , 1 , 5 ] , [ 2 , 2 , 4 , 1 ] , [ 5 , 0 , 2 , 3 ] , [ 0 , 6 , 1 , 2 ] ] NEW_LINE m = 4 NEW_LINE n = 4 NEW_LINE print ( getMaxGold ( gold , m , n ) ) NEW_LINE
M = 100 NEW_LINE
def minAdjustmentCost ( A , n , target ) : NEW_LINE
dp = [ [ 0 for i in range ( M + 1 ) ] for i in range ( n ) ] NEW_LINE
for j in range ( M + 1 ) : NEW_LINE INDENT dp [ 0 ] [ j ] = abs ( j - A [ 0 ] ) NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE
for j in range ( M + 1 ) : NEW_LINE
dp [ i ] [ j ] = 100000000 NEW_LINE
for k in range ( max ( j - target , 0 ) , min ( M , j + target ) + 1 ) : NEW_LINE INDENT dp [ i ] [ j ] = min ( dp [ i ] [ j ] , dp [ i - 1 ] [ k ] + abs ( A [ i ] - j ) ) NEW_LINE DEDENT
res = 10000000 NEW_LINE for j in range ( M + 1 ) : NEW_LINE INDENT res = min ( res , dp [ n - 1 ] [ j ] ) NEW_LINE DEDENT return res NEW_LINE
arr = [ 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 ] NEW_LINE n = len ( arr ) NEW_LINE target = 10 NEW_LINE print ( " Minimum ▁ adjustment ▁ cost ▁ is " , minAdjustmentCost ( arr , n , target ) , sep = ' ▁ ' ) NEW_LINE
' NEW_LINE def countChar ( str , x ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( len ( str ) ) : NEW_LINE INDENT if ( str [ i ] == x ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT n = 10 NEW_LINE DEDENT
repetitions = n // len ( str ) NEW_LINE count = count * repetitions NEW_LINE
l = n % len ( str ) NEW_LINE for i in range ( l ) : NEW_LINE INDENT if ( str [ i ] == x ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
str = " abcac " NEW_LINE print ( countChar ( str , ' a ' ) ) NEW_LINE
def check ( s , m ) : NEW_LINE
l = len ( s ) ; NEW_LINE
c1 = 0 ; NEW_LINE
c2 = 0 ; NEW_LINE for i in range ( 0 , l - 1 ) : NEW_LINE INDENT if ( s [ i ] == '0' ) : NEW_LINE INDENT c2 = 0 ; NEW_LINE DEDENT DEDENT
c1 = c1 + 1 ; NEW_LINE else : NEW_LINE c1 = 0 ; NEW_LINE
c2 = c2 + 1 ; NEW_LINE if ( c1 == m or c2 == m ) : NEW_LINE return True ; NEW_LINE return False ; NEW_LINE
s = "001001" ; NEW_LINE m = 2 ; NEW_LINE
if ( check ( s , m ) ) : NEW_LINE INDENT print ( " YES " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) ; NEW_LINE DEDENT
def productAtKthLevel ( tree , k ) : NEW_LINE INDENT level = - 1 NEW_LINE DEDENT
product = 1 NEW_LINE n = len ( tree ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if ( tree [ i ] == ' ( ' ) : NEW_LINE INDENT level += 1 NEW_LINE DEDENT
elif ( tree [ i ] == ' ) ' ) : NEW_LINE INDENT level -= 1 NEW_LINE DEDENT else : NEW_LINE
if ( level == k ) : NEW_LINE INDENT product *= ( int ( tree [ i ] ) - int ( '0' ) ) NEW_LINE DEDENT
return product NEW_LINE
tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " NEW_LINE k = 2 NEW_LINE print ( productAtKthLevel ( tree , k ) ) NEW_LINE
def isValidISBN ( isbn ) : NEW_LINE
if len ( isbn ) != 10 : NEW_LINE INDENT return False NEW_LINE DEDENT
_sum = 0 NEW_LINE for i in range ( 9 ) : NEW_LINE INDENT if 0 <= int ( isbn [ i ] ) <= 9 : NEW_LINE INDENT _sum += int ( isbn [ i ] ) * ( 10 - i ) NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
if ( isbn [ 9 ] != ' X ' and 0 <= int ( isbn [ 9 ] ) <= 9 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
_sum += 10 if isbn [ 9 ] == ' X ' else int ( isbn [ 9 ] ) NEW_LINE
return ( _sum % 11 == 0 ) NEW_LINE
isbn = "007462542X " NEW_LINE if isValidISBN ( isbn ) : NEW_LINE INDENT print ( ' Valid ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Invalid " ) NEW_LINE DEDENT
d = 10 NEW_LINE a = 0.0 NEW_LINE
a = ( 360 - ( 6 * d ) ) / 4 NEW_LINE
print ( a , " , " , a + d , " , " , a + 2 * d , " , " , a + 3 * d , sep = ' ▁ ' ) NEW_LINE
def distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) : NEW_LINE INDENT if ( a1 / a2 == b1 / b2 and b1 / b2 == c1 / c2 ) : NEW_LINE INDENT x1 = y1 = 0 NEW_LINE z1 = - d1 / c1 NEW_LINE d = abs ( ( c2 * z1 + d2 ) ) / ( math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " ) , d NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Planes ▁ are ▁ not ▁ parallel " ) NEW_LINE DEDENT DEDENT
a1 = 1 NEW_LINE b1 = 2 NEW_LINE c1 = - 1 NEW_LINE d1 = 1 NEW_LINE a2 = 3 NEW_LINE b2 = 6 NEW_LINE c2 = - 3 NEW_LINE d2 = - 4 NEW_LINE distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) NEW_LINE
def PrintMinNumberForPattern ( arr ) : NEW_LINE
curr_max = 0 NEW_LINE
last_entry = 0 NEW_LINE i = 0 NEW_LINE
while i < len ( arr ) : NEW_LINE
noOfNextD = 0 NEW_LINE if arr [ i ] == " I " : NEW_LINE
j = i + 1 NEW_LINE while j < len ( arr ) and arr [ j ] == " D " : NEW_LINE INDENT noOfNextD += 1 NEW_LINE j += 1 NEW_LINE DEDENT if i == 0 : NEW_LINE INDENT curr_max = noOfNextD + 2 NEW_LINE last_entry += 1 NEW_LINE DEDENT
print ( " " , last_entry , end = " " ) NEW_LINE print ( " " , curr_max , end = " " ) NEW_LINE
last_entry = curr_max NEW_LINE else : NEW_LINE
curr_max += noOfNextD + 1 NEW_LINE
last_entry = curr_max NEW_LINE print ( " " , last_entry , end = " " ) NEW_LINE
for k in range ( noOfNextD ) : NEW_LINE INDENT last_entry -= 1 NEW_LINE print ( " " , last_entry , end = " " ) NEW_LINE i += 1 NEW_LINE DEDENT
elif arr [ i ] == " D " : NEW_LINE INDENT if i == 0 : NEW_LINE DEDENT
j = i + 1 NEW_LINE while j < len ( arr ) and arr [ j ] == " D " : NEW_LINE INDENT noOfNextD += 1 NEW_LINE j += 1 NEW_LINE DEDENT
curr_max = noOfNextD + 2 NEW_LINE
print ( " " , curr_max , curr_max - 1 , end = " " ) NEW_LINE
last_entry = curr_max - 1 NEW_LINE else : NEW_LINE
print ( " " , last_entry - 1 , end = " " ) NEW_LINE last_entry -= 1 NEW_LINE i += 1 NEW_LINE print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT PrintMinNumberForPattern ( " IDID " ) NEW_LINE PrintMinNumberForPattern ( " I " ) NEW_LINE PrintMinNumberForPattern ( " DD " ) NEW_LINE PrintMinNumberForPattern ( " II " ) NEW_LINE PrintMinNumberForPattern ( " DIDI " ) NEW_LINE PrintMinNumberForPattern ( " IIDDD " ) NEW_LINE PrintMinNumberForPattern ( " DDIDDIID " ) NEW_LINE DEDENT
def isPrime ( n ) : NEW_LINE INDENT i , c = 0 , 0 NEW_LINE for i in range ( 1 , n // 2 ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT if ( c == 1 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
def findMinNum ( arr , n ) : NEW_LINE
first , last = 0 , 0 NEW_LINE Hash = [ 0 for i in range ( 10 ) ] NEW_LINE
for i in range ( n ) : NEW_LINE INDENT Hash [ arr [ i ] ] += 1 NEW_LINE DEDENT
print ( " Minimum ▁ number : ▁ " , end = " " ) NEW_LINE for i in range ( 0 , 10 ) : NEW_LINE
for j in range ( Hash [ i ] ) : NEW_LINE INDENT print ( i , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE
for i in range ( 10 ) : NEW_LINE INDENT if ( Hash [ i ] != 0 ) : NEW_LINE INDENT first = i NEW_LINE break NEW_LINE DEDENT DEDENT
for i in range ( 9 , - 1 , - 1 ) : NEW_LINE INDENT if ( Hash [ i ] != 0 ) : NEW_LINE INDENT last = i NEW_LINE break NEW_LINE DEDENT DEDENT num = first * 10 + last NEW_LINE rev = last * 10 + first NEW_LINE
print ( " Prime ▁ combinations : ▁ " , end = " " ) NEW_LINE if ( isPrime ( num ) and isPrime ( rev ) ) : NEW_LINE INDENT print ( num , " ▁ " , rev ) NEW_LINE DEDENT elif ( isPrime ( num ) ) : NEW_LINE INDENT print ( num ) NEW_LINE DEDENT elif ( isPrime ( rev ) ) : NEW_LINE INDENT print ( rev ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ combinations ▁ exist " ) NEW_LINE DEDENT
arr = [ 1 , 2 , 4 , 7 , 8 ] NEW_LINE findMinNum ( arr , 5 ) NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT return gcd ( b % a , a ) ; NEW_LINE DEDENT
def coprime ( a , b ) : NEW_LINE
return ( gcd ( a , b ) == 1 ) ; NEW_LINE
def possibleTripletInRange ( L , R ) : NEW_LINE INDENT flag = False ; NEW_LINE possibleA = 0 ; NEW_LINE possibleB = 0 ; NEW_LINE possibleC = 0 ; NEW_LINE DEDENT
for a in range ( L , R + 1 ) : NEW_LINE INDENT for b in range ( a + 1 , R + 1 ) : NEW_LINE INDENT for c in range ( b + 1 , R + 1 ) : NEW_LINE DEDENT DEDENT
if ( coprime ( a , b ) and coprime ( b , c ) and coprime ( a , c ) == False ) : NEW_LINE INDENT flag = True ; NEW_LINE possibleA = a ; NEW_LINE possibleB = b ; NEW_LINE possibleC = c ; NEW_LINE break ; NEW_LINE DEDENT
if ( flag == True ) : NEW_LINE INDENT print ( " ( " , possibleA , " , " , possibleB , " , " , possibleC , " ) ▁ is ▁ one ▁ such " , " possible ▁ triplet ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Such ▁ Triplet ▁ exists ▁ between " , L , " and " , R ) ; NEW_LINE DEDENT
L = 2 ; NEW_LINE R = 10 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
L = 23 ; NEW_LINE R = 46 ; NEW_LINE possibleTripletInRange ( L , R ) ; NEW_LINE
def possibleToReach ( a , b ) : NEW_LINE
c = np . cbrt ( a * b ) NEW_LINE
re1 = a // c NEW_LINE re2 = b // c NEW_LINE
if ( ( re1 * re1 * re2 == a ) and ( re2 * re2 * re1 == b ) ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = 60 NEW_LINE B = 450 NEW_LINE if ( possibleToReach ( A , B ) ) : NEW_LINE INDENT print ( " yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " no " ) NEW_LINE DEDENT DEDENT
def isUndulating ( n ) : NEW_LINE
if ( len ( n ) <= 2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , len ( n ) ) : NEW_LINE INDENT if ( n [ i - 2 ] != n [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
n = "1212121" NEW_LINE if ( isUndulating ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def Series ( n ) : NEW_LINE INDENT sums = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT sums += ( i * i ) ; NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = Series ( n ) NEW_LINE print ( res ) NEW_LINE
sdef sumDivisible ( L , R ) : NEW_LINE
p = int ( R / 6 ) NEW_LINE
q = int ( ( L - 1 ) / 6 ) NEW_LINE
sumR = 3 * ( p * ( p + 1 ) ) NEW_LINE
sumL = ( q * ( q + 1 ) ) * 3 NEW_LINE
return sumR - sumL NEW_LINE
L = 1 NEW_LINE R = 20 NEW_LINE print ( sumDivisible ( L , R ) ) NEW_LINE
import sys NEW_LINE
def prevNum ( string , n ) : NEW_LINE INDENT index = - 1 NEW_LINE DEDENT
for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT if int ( string [ i ] ) > int ( string [ i + 1 ] ) : NEW_LINE INDENT index = i NEW_LINE break NEW_LINE DEDENT DEDENT
smallGreatDgt = - 1 NEW_LINE for i in range ( n - 1 , index , - 1 ) : NEW_LINE INDENT if ( smallGreatDgt == - 1 and int ( string [ i ] ) < int ( string [ index ] ) ) : NEW_LINE INDENT smallGreatDgt = i NEW_LINE DEDENT elif ( index > - 1 and int ( string [ i ] ) >= int ( string [ smallGreatDgt ] ) and int ( string [ i ] ) < int ( string [ index ] ) ) : NEW_LINE INDENT smallGreatDgt = i NEW_LINE DEDENT DEDENT
if index == - 1 : NEW_LINE INDENT return " " . join ( " - 1" ) NEW_LINE DEDENT else : NEW_LINE
( string [ index ] , string [ smallGreatDgt ] ) = ( string [ smallGreatDgt ] , string [ index ] ) NEW_LINE return " " . join ( string ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n_str = "34125" NEW_LINE ans = prevNum ( list ( n_str ) , len ( n_str ) ) NEW_LINE print ( ans ) NEW_LINE DEDENT
def horner ( poly , n , x ) : NEW_LINE
result = poly [ 0 ] ; NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT result = ( result * x + poly [ i ] ) ; NEW_LINE DEDENT return result ; NEW_LINE
def findSign ( poly , n , x ) : NEW_LINE INDENT result = horner ( poly , n , x ) ; NEW_LINE if ( result > 0 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT elif ( result < 0 ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT return 0 ; NEW_LINE DEDENT
poly = [ 2 , - 6 , 2 , - 1 ] ; NEW_LINE x = 3 ; NEW_LINE n = len ( poly ) ; NEW_LINE print ( " Sign ▁ of ▁ polynomial ▁ is ▁ " , findSign ( poly , n , x ) ) ; NEW_LINE
isPrime = [ 1 ] * 100005 NEW_LINE
def findPrime ( n ) : NEW_LINE INDENT num = n + 1 NEW_LINE DEDENT
while ( num ) : NEW_LINE
if isPrime [ num ] : NEW_LINE INDENT return num NEW_LINE DEDENT
num += 1 NEW_LINE return 0 NEW_LINE
def minNumber ( arr ) : NEW_LINE
sieveOfEratostheneses ( ) NEW_LINE s = 0 NEW_LINE
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT s += arr [ i ] NEW_LINE DEDENT
if isPrime [ s ] == True : NEW_LINE INDENT return 0 NEW_LINE DEDENT
num = findPrime ( s ) NEW_LINE
return num - s NEW_LINE
arr = [ 2 , 4 , 6 , 8 , 12 ] NEW_LINE print ( minNumber ( arr ) ) NEW_LINE
def SubArraySum ( arr , n ) : NEW_LINE INDENT temp , result = 0 , 0 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE
temp = 0 ; NEW_LINE for j in range ( i , n ) : NEW_LINE
temp += arr [ j ] NEW_LINE result += temp NEW_LINE return result NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Sum ▁ of ▁ SubArray ▁ : " , SubArraySum ( arr , n ) ) NEW_LINE
import math NEW_LINE def highestPowerof2 ( n ) : NEW_LINE INDENT p = int ( math . log ( n , 2 ) ) ; NEW_LINE return int ( pow ( 2 , p ) ) ; NEW_LINE DEDENT
n = 10 ; NEW_LINE print ( highestPowerof2 ( n ) ) ; NEW_LINE
def aModM ( s , mod ) : NEW_LINE INDENT number = 0 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE DEDENT
number = ( number * 10 + int ( s [ i ] ) ) NEW_LINE number = number % m NEW_LINE return number NEW_LINE
def ApowBmodM ( a , b , m ) : NEW_LINE
ans = aModM ( a , m ) NEW_LINE mul = ans NEW_LINE
for i in range ( 1 , b ) : NEW_LINE INDENT ans = ( ans * mul ) % m NEW_LINE DEDENT return ans NEW_LINE
a = "987584345091051645734583954832576" NEW_LINE b , m = 3 , 11 NEW_LINE print ApowBmodM ( a , b , m ) NEW_LINE
def SieveOfSundaram ( n ) : NEW_LINE
nNew = int ( ( n - 1 ) / 2 ) ; NEW_LINE
for i in range ( 1 , nNew + 1 ) : NEW_LINE INDENT j = i ; NEW_LINE while ( ( i + j + 2 * i * j ) <= nNew ) : NEW_LINE INDENT marked [ i + j + 2 * i * j ] = 1 ; NEW_LINE j += 1 ; NEW_LINE DEDENT DEDENT
if ( n > 2 ) : NEW_LINE INDENT print ( 2 , end = " ▁ " ) ; NEW_LINE DEDENT
for i in range ( 1 , nNew + 1 ) : NEW_LINE INDENT if ( marked [ i ] == 0 ) : NEW_LINE INDENT print ( ( 2 * i + 1 ) , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT
n = 20 ; NEW_LINE SieveOfSundaram ( n ) ; NEW_LINE
def hammingDistance ( n1 , n2 ) : NEW_LINE INDENT x = n1 ^ n2 NEW_LINE setBits = 0 NEW_LINE while ( x > 0 ) : NEW_LINE INDENT setBits += x & 1 NEW_LINE x >>= 1 NEW_LINE DEDENT return setBits NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n1 = 9 NEW_LINE n2 = 14 NEW_LINE print ( hammingDistance ( 9 , 14 ) ) NEW_LINE DEDENT
def printSubsets ( n ) : NEW_LINE INDENT for i in range ( n + 1 ) : NEW_LINE INDENT if ( ( n & i ) == i ) : NEW_LINE INDENT print ( i , " ▁ " , end = " " ) NEW_LINE DEDENT DEDENT DEDENT
n = 9 NEW_LINE printSubsets ( n ) NEW_LINE
import math NEW_LINE def setBitNumber ( n ) : NEW_LINE
k = int ( math . log ( n , 2 ) ) NEW_LINE
return 1 << k NEW_LINE
n = 273 NEW_LINE print ( setBitNumber ( n ) ) NEW_LINE
def subset ( ar , n ) : NEW_LINE
res = 0 NEW_LINE
ar . sort ( ) NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT count = 1 NEW_LINE DEDENT
for i in range ( n - 1 ) : NEW_LINE INDENT if ar [ i ] == ar [ i + 1 ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT
res = max ( res , count ) NEW_LINE return res NEW_LINE
ar = [ 5 , 6 , 9 , 3 , 4 , 3 , 4 ] NEW_LINE n = len ( ar ) NEW_LINE print ( subset ( ar , n ) ) NEW_LINE
def areElementsContiguous ( arr , n ) : NEW_LINE
arr . sort ( ) NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT if ( arr [ i ] - arr [ i - 1 ] > 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT return 1 NEW_LINE
arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE if areElementsContiguous ( arr , n ) : print ( " Yes " ) NEW_LINE else : print ( " No " ) NEW_LINE
def findLargestd ( S , n ) : NEW_LINE INDENT found = False NEW_LINE DEDENT
S . sort ( ) NEW_LINE
for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE INDENT for j in range ( 0 , n ) : NEW_LINE DEDENT
if ( i == j ) : NEW_LINE INDENT continue NEW_LINE DEDENT for k in range ( j + 1 , n ) : NEW_LINE INDENT if ( i == k ) : NEW_LINE INDENT continue NEW_LINE DEDENT for l in range ( k + 1 , n ) : NEW_LINE INDENT if ( i == l ) : NEW_LINE INDENT continue NEW_LINE DEDENT DEDENT DEDENT
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) : NEW_LINE INDENT found = True NEW_LINE return S [ i ] NEW_LINE DEDENT if ( found == False ) : NEW_LINE return - 1 NEW_LINE
S = [ 2 , 3 , 5 , 7 , 12 ] NEW_LINE n = len ( S ) NEW_LINE ans = findLargestd ( S , n ) NEW_LINE if ( ans == - 1 ) : NEW_LINE INDENT print ( " No ▁ Solution " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ b ▁ + " , " c ▁ = ▁ d ▁ is " , ans ) NEW_LINE DEDENT
def leftRotatebyOne ( arr , n ) : NEW_LINE INDENT temp = arr [ 0 ] NEW_LINE for i in range ( n - 1 ) : NEW_LINE INDENT arr [ i ] = arr [ i + 1 ] NEW_LINE DEDENT arr [ n - 1 ] = temp NEW_LINE DEDENT
def leftRotate ( arr , d , n ) : NEW_LINE INDENT for i in range ( d ) : NEW_LINE INDENT leftRotatebyOne ( arr , n ) NEW_LINE DEDENT DEDENT
def printArray ( arr , size ) : NEW_LINE INDENT for i in range ( size ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE leftRotate ( arr , 2 , 7 ) NEW_LINE printArray ( arr , 7 ) NEW_LINE
def partSort ( arr , N , a , b ) : NEW_LINE
l = min ( a , b ) NEW_LINE r = max ( a , b ) NEW_LINE
temp = [ 0 for i in range ( r - l + 1 ) ] NEW_LINE j = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE INDENT temp [ j ] = arr [ i ] NEW_LINE j += 1 NEW_LINE DEDENT
temp . sort ( reverse = False ) NEW_LINE
j = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE INDENT arr [ i ] = temp [ j ] NEW_LINE j += 1 NEW_LINE DEDENT
for i in range ( 0 , N , 1 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 7 , 8 , 4 , 5 , 2 ] NEW_LINE a = 1 NEW_LINE b = 4 NEW_LINE DEDENT
N = len ( arr ) NEW_LINE partSort ( arr , N , a , b ) NEW_LINE
def pushZerosToEnd ( arr , n ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if arr [ i ] != 0 : NEW_LINE DEDENT
arr [ count ] = arr [ i ] NEW_LINE count += 1 NEW_LINE
while count < n : NEW_LINE INDENT arr [ count ] = 0 NEW_LINE count += 1 NEW_LINE DEDENT
arr = [ 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE pushZerosToEnd ( arr , n ) NEW_LINE print ( " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ end ▁ of ▁ array : " ) NEW_LINE print ( arr ) NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def RearrangePosNeg ( arr , n ) : NEW_LINE INDENT for i in range ( 1 , n ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE DEDENT DEDENT
if ( key > 0 ) : NEW_LINE INDENT continue NEW_LINE DEDENT
j = i - 1 NEW_LINE while ( j >= 0 and arr [ j ] > 0 ) : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT
arr [ j + 1 ] = key NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] NEW_LINE n = len ( arr ) NEW_LINE RearrangePosNeg ( arr , n ) NEW_LINE printArray ( arr , n ) NEW_LINE DEDENT
def findElements ( arr , n ) : NEW_LINE
for i in range ( n ) : NEW_LINE INDENT count = 0 NEW_LINE for j in range ( 0 , n ) : NEW_LINE INDENT if arr [ j ] > arr [ i ] : NEW_LINE INDENT count = count + 1 NEW_LINE DEDENT DEDENT if count >= 2 : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
def findElements ( arr , n ) : NEW_LINE INDENT arr . sort ( ) NEW_LINE for i in range ( 0 , n - 2 ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
import sys NEW_LINE def findElements ( arr , n ) : NEW_LINE INDENT first = - sys . maxsize NEW_LINE second = - sys . maxsize NEW_LINE for i in range ( 0 , n ) : NEW_LINE DEDENT
if ( arr [ i ] > first ) : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] > second ) : NEW_LINE INDENT second = arr [ i ] NEW_LINE DEDENT for i in range ( 0 , n ) : NEW_LINE if ( arr [ i ] < second ) : NEW_LINE INDENT print ( arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
arr = [ 2 , - 6 , 3 , 5 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE findElements ( arr , n ) NEW_LINE
def findFirstMissing ( array , start , end ) : NEW_LINE INDENT if ( start > end ) : NEW_LINE INDENT return end + 1 NEW_LINE DEDENT if ( start != array [ start ] ) : NEW_LINE INDENT return start ; NEW_LINE DEDENT mid = int ( ( start + end ) / 2 ) NEW_LINE DEDENT
if ( array [ mid ] == mid ) : NEW_LINE INDENT return findFirstMissing ( array , mid + 1 , end ) NEW_LINE DEDENT return findFirstMissing ( array , start , mid ) NEW_LINE
arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Smallest ▁ missing ▁ element ▁ is " , findFirstMissing ( arr , 0 , n - 1 ) ) NEW_LINE
def find_max_sum ( arr ) : NEW_LINE INDENT incl = 0 NEW_LINE excl = 0 NEW_LINE for i in arr : NEW_LINE DEDENT
new_excl = excl if excl > incl else incl NEW_LINE
incl = excl + i NEW_LINE excl = new_excl NEW_LINE
return ( excl if excl > incl else incl ) NEW_LINE
arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] NEW_LINE print find_max_sum ( arr ) NEW_LINE
def findMaxAverage ( arr , n , k ) : NEW_LINE
if k > n : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
csum = [ 0 ] * n NEW_LINE csum [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT csum [ i ] = csum [ i - 1 ] + arr [ i ] ; NEW_LINE DEDENT
max_sum = csum [ k - 1 ] NEW_LINE max_end = k - 1 NEW_LINE
for i in range ( k , n ) : NEW_LINE INDENT curr_sum = csum [ i ] - csum [ i - k ] NEW_LINE if curr_sum > max_sum : NEW_LINE INDENT max_sum = curr_sum NEW_LINE max_end = i NEW_LINE DEDENT DEDENT
return max_end - k + 1 NEW_LINE
arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] NEW_LINE k = 4 NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ length " , k , " begins ▁ at ▁ index " , findMaxAverage ( arr , n , k ) ) NEW_LINE
def findMaxAverage ( arr , n , k ) : NEW_LINE
if ( k > n ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
sum = arr [ 0 ] NEW_LINE for i in range ( 1 , k ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT max_sum = sum NEW_LINE max_end = k - 1 NEW_LINE
for i in range ( k , n ) : NEW_LINE INDENT sum = sum + arr [ i ] - arr [ i - k ] NEW_LINE if ( sum > max_sum ) : NEW_LINE INDENT max_sum = sum NEW_LINE max_end = i NEW_LINE DEDENT DEDENT
return max_end - k + 1 NEW_LINE
arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] NEW_LINE k = 4 NEW_LINE n = len ( arr ) NEW_LINE print ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ length " , k , " begins ▁ at ▁ index " , findMaxAverage ( arr , n , k ) ) NEW_LINE
def isMajority ( arr , n , x ) : NEW_LINE
last_index = ( n // 2 + 1 ) if n % 2 == 0 else ( n // 2 ) NEW_LINE
for i in range ( last_index ) : NEW_LINE
if arr [ i ] == x and arr [ i + n // 2 ] == x : NEW_LINE INDENT return 1 NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE x = 4 NEW_LINE if ( isMajority ( arr , n , x ) ) : NEW_LINE INDENT print ( " % ▁ d ▁ appears ▁ more ▁ than ▁ % ▁ d ▁ times ▁ in ▁ arr [ ] " % ( x , n // 2 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " % ▁ d ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ % ▁ d ▁ times ▁ in ▁ arr [ ] " % ( x , n // 2 ) ) NEW_LINE DEDENT
def cutRod ( price , n ) : NEW_LINE INDENT val = [ 0 for x in range ( n + 1 ) ] NEW_LINE val [ 0 ] = 0 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT max_val = INT_MIN NEW_LINE for j in range ( i ) : NEW_LINE INDENT max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) NEW_LINE DEDENT val [ i ] = max_val NEW_LINE DEDENT return val [ n ] NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + str ( cutRod ( arr , size ) ) ) NEW_LINE
def primeCount ( arr , n ) : NEW_LINE
max_val = max ( arr ) NEW_LINE
prime = [ True ] * ( max_val + 1 ) NEW_LINE
prime [ 0 ] = prime [ 1 ] = False NEW_LINE p = 2 NEW_LINE while p * p <= max_val : NEW_LINE
if prime [ p ] == True : NEW_LINE
for i in range ( p * 2 , max_val + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT if prime [ arr [ i ] ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE
def getPrefixArray ( arr , n , pre ) : NEW_LINE
pre [ 0 ] = arr [ 0 ] NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT pre [ i ] = pre [ i - 1 ] + arr [ i ] NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 4 , 8 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE DEDENT
pre = [ None ] * n NEW_LINE getPrefixArray ( arr , n , pre ) NEW_LINE
print ( primeCount ( pre , n ) ) NEW_LINE
def minValue ( n , x , y ) : NEW_LINE
val = ( y * n ) / 100 NEW_LINE
if x >= val : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE INDENT return math . ceil ( val ) - x NEW_LINE DEDENT
n = 10 ; x = 2 ; y = 40 NEW_LINE print ( minValue ( n , x , y ) ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT for i in range ( 5 , int ( sqrt ( n ) ) + 1 , 6 ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isFactorialPrime ( n ) : NEW_LINE
if ( not isPrime ( n ) ) : NEW_LINE INDENT return False NEW_LINE DEDENT fact = 1 NEW_LINE i = 1 NEW_LINE while ( fact <= n + 1 ) : NEW_LINE
fact = fact * i NEW_LINE
if ( n + 1 == fact or n - 1 == fact ) : NEW_LINE INDENT return True NEW_LINE DEDENT i += 1 NEW_LINE
return False NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 23 NEW_LINE if ( isFactorialPrime ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
n = 5 NEW_LINE
fac1 = 1 NEW_LINE for i in range ( 2 , n ) : NEW_LINE INDENT fac1 = fac1 * i NEW_LINE DEDENT
fac2 = fac1 * n NEW_LINE
totalWays = fac1 * fac2 NEW_LINE
print ( totalWays ) NEW_LINE
def nextPerfectCube ( N ) : NEW_LINE INDENT nextN = floor ( N ** ( 1 / 3 ) ) + 1 NEW_LINE return nextN ** 3 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 35 NEW_LINE print ( nextPerfectCube ( n ) ) NEW_LINE DEDENT
def findpos ( n ) : NEW_LINE INDENT pos = 0 NEW_LINE for i in n : NEW_LINE DEDENT
if i == '2' : NEW_LINE INDENT pos = pos * 4 + 1 NEW_LINE DEDENT
elif i == '3' : NEW_LINE INDENT pos = pos * 4 + 2 NEW_LINE DEDENT
elif i == '5' : NEW_LINE INDENT pos = pos * 4 + 3 NEW_LINE DEDENT
elif i == '7' : NEW_LINE INDENT pos = pos * 4 + 4 NEW_LINE DEDENT return pos NEW_LINE
n = "777" NEW_LINE print ( findpos ( n ) ) NEW_LINE
mod = 1000000007 NEW_LINE
def digitNumber ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n == 1 ) : NEW_LINE INDENT return 9 NEW_LINE DEDENT
if ( n % 2 != 0 ) : NEW_LINE
temp = digitNumber ( ( n - 1 ) // 2 ) % mod NEW_LINE return ( 9 * ( temp * temp ) % mod ) % mod NEW_LINE else : NEW_LINE
temp = digitNumber ( n // 2 ) % mod NEW_LINE return ( temp * temp ) % mod NEW_LINE def countExcluding ( n , d ) : NEW_LINE
if ( d == 0 ) : NEW_LINE INDENT return ( 9 * digitNumber ( n - 1 ) ) % mod NEW_LINE DEDENT else : NEW_LINE INDENT return ( 8 * digitNumber ( n - 1 ) ) % mod NEW_LINE DEDENT
d = 9 NEW_LINE n = 3 NEW_LINE print ( countExcluding ( n , d ) ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , n ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isEmirp ( n ) : NEW_LINE
n = int ( n ) NEW_LINE if isPrime ( n ) == False : NEW_LINE INDENT return False NEW_LINE DEDENT
rev = 0 NEW_LINE while n != 0 : NEW_LINE INDENT d = n % 10 NEW_LINE rev = rev * 10 + d NEW_LINE n = int ( n / 10 ) NEW_LINE DEDENT
return isPrime ( rev ) NEW_LINE
n = 13 NEW_LINE if isEmirp ( n ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def Convert ( radian ) : NEW_LINE INDENT pi = 3.14159 NEW_LINE degree = radian * ( 180 / pi ) NEW_LINE return degree NEW_LINE DEDENT
radian = 5 NEW_LINE print ( " degree ▁ = " , ( Convert ( radian ) ) ) NEW_LINE
def sn ( n , an ) : NEW_LINE INDENT return ( n * ( 1 + an ) ) / 2 ; NEW_LINE DEDENT
def trace ( n , m ) : NEW_LINE
an = 1 + ( n - 1 ) * ( m + 1 ) ; NEW_LINE
rowmajorSum = sn ( n , an ) ; NEW_LINE
an = 1 + ( n - 1 ) * ( n + 1 ) ; NEW_LINE
colmajorSum = sn ( n , an ) ; NEW_LINE return int ( rowmajorSum + colmajorSum ) ; NEW_LINE
N = 3 ; NEW_LINE M = 3 ; NEW_LINE print ( trace ( N , M ) ) ; NEW_LINE
def max_area ( n , m , k ) : NEW_LINE INDENT if ( k > ( n + m - 2 ) ) : NEW_LINE INDENT print ( " Not ▁ possible " ) NEW_LINE DEDENT else : NEW_LINE DEDENT
if ( k < max ( m , n ) - 1 ) : NEW_LINE INDENT result = max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT result = max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; NEW_LINE DEDENT
print ( result ) NEW_LINE
n = 3 NEW_LINE m = 4 NEW_LINE k = 1 NEW_LINE max_area ( n , m , k ) NEW_LINE
def area_fun ( side ) : NEW_LINE INDENT area = side * side NEW_LINE return area NEW_LINE DEDENT
side = 4 NEW_LINE area = area_fun ( side ) NEW_LINE print ( area ) NEW_LINE
def countConsecutive ( N ) : NEW_LINE
count = 0 NEW_LINE L = 1 NEW_LINE while ( L * ( L + 1 ) < 2 * N ) : NEW_LINE INDENT a = ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) NEW_LINE if ( a - int ( a ) == 0.0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT L += 1 NEW_LINE DEDENT return count NEW_LINE
N = 15 NEW_LINE print countConsecutive ( N ) NEW_LINE N = 10 NEW_LINE print countConsecutive ( N ) NEW_LINE
def isAutomorphic ( N ) : NEW_LINE
sq = N * N NEW_LINE
while ( N > 0 ) : NEW_LINE
if ( N % 10 != sq % 10 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
N /= 10 NEW_LINE sq /= 10 NEW_LINE return True NEW_LINE
N = 5 NEW_LINE if isAutomorphic ( N ) : NEW_LINE INDENT print " Automorphic " NEW_LINE DEDENT else : NEW_LINE INDENT print " Not ▁ Automorphic " NEW_LINE DEDENT
def maxPrimefactorNum ( N ) : NEW_LINE
arr = [ True ] * ( N + 5 ) ; NEW_LINE
i = 3 ; NEW_LINE while ( i * i <= N ) : NEW_LINE INDENT if ( arr [ i ] ) : NEW_LINE INDENT for j in range ( i * i , N + 1 , i ) : NEW_LINE INDENT arr [ j ] = False ; NEW_LINE DEDENT DEDENT i += 2 ; NEW_LINE DEDENT
prime = [ ] ; NEW_LINE prime . append ( 2 ) ; NEW_LINE for i in range ( 3 , N + 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] ) : NEW_LINE INDENT prime . append ( i ) ; NEW_LINE DEDENT DEDENT
i = 0 ; NEW_LINE ans = 1 ; NEW_LINE while ( ans * prime [ i ] <= N and i < len ( prime ) ) : NEW_LINE INDENT ans *= prime [ i ] ; NEW_LINE i += 1 ; NEW_LINE DEDENT return ans ; NEW_LINE
N = 40 ; NEW_LINE print ( maxPrimefactorNum ( N ) ) ; NEW_LINE
def divSum ( num ) : NEW_LINE
result = 0 NEW_LINE
' NEW_LINE INDENT i = 2 NEW_LINE while i <= ( math . sqrt ( num ) ) : NEW_LINE DEDENT
' NEW_LINE INDENT if ( num % i == 0 ) : NEW_LINE DEDENT
if ( i == ( num / i ) ) : NEW_LINE INDENT result = result + i ; NEW_LINE DEDENT else : NEW_LINE INDENT result = result + ( i + num / i ) ; NEW_LINE DEDENT i = i + 1 NEW_LINE
return ( result + 1 ) ; NEW_LINE
num = 36 NEW_LINE print ( divSum ( num ) ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
x = ( x * x ) % p NEW_LINE return res NEW_LINE
def squareRoot ( n , p ) : NEW_LINE INDENT if ( p % 4 != 3 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return NEW_LINE DEDENT DEDENT
n = n % p NEW_LINE x = power ( n , ( p + 1 ) // 4 , p ) NEW_LINE if ( ( x * x ) % p == n ) : NEW_LINE INDENT print ( " Square ▁ root ▁ is ▁ " , x ) NEW_LINE return NEW_LINE DEDENT
x = p - x NEW_LINE if ( ( x * x ) % p == n ) : NEW_LINE INDENT print ( " Square ▁ root ▁ is ▁ " , x ) NEW_LINE return NEW_LINE DEDENT
print ( " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ) NEW_LINE
p = 7 NEW_LINE n = 2 NEW_LINE squareRoot ( n , p ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
res = 1 ; NEW_LINE
x = x % p ; NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p ; NEW_LINE DEDENT
x = ( x * x ) % p ; NEW_LINE return res ; NEW_LINE
def miillerTest ( d , n ) : NEW_LINE
a = 2 + random . randint ( 1 , n - 4 ) ; NEW_LINE
x = power ( a , d , n ) ; NEW_LINE if ( x == 1 or x == n - 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
while ( d != n - 1 ) : NEW_LINE INDENT x = ( x * x ) % n ; NEW_LINE d *= 2 ; NEW_LINE if ( x == 1 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT if ( x == n - 1 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT DEDENT
return False ; NEW_LINE
def isPrime ( n , k ) : NEW_LINE
if ( n <= 1 or n == 4 ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT
d = n - 1 ; NEW_LINE while ( d % 2 == 0 ) : NEW_LINE INDENT d //= 2 ; NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT if ( miillerTest ( d , n ) == False ) : NEW_LINE INDENT return False ; NEW_LINE DEDENT DEDENT return True ; NEW_LINE
k = 4 ; NEW_LINE print ( " All ▁ primes ▁ smaller ▁ than ▁ 100 : ▁ " ) ; NEW_LINE for n in range ( 1 , 100 ) : NEW_LINE INDENT if ( isPrime ( n , k ) ) : NEW_LINE INDENT print ( n , end = " ▁ " ) ; NEW_LINE DEDENT DEDENT
def maxConsecutiveOnes ( x ) : NEW_LINE
count = 0 NEW_LINE
while ( x != 0 ) : NEW_LINE
x = ( x & ( x << 1 ) ) NEW_LINE count = count + 1 NEW_LINE return count NEW_LINE
print ( maxConsecutiveOnes ( 14 ) ) NEW_LINE print ( maxConsecutiveOnes ( 222 ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE
while ( y != 0 ) : NEW_LINE
borrow = ( ~ x ) & y NEW_LINE
x = x ^ y NEW_LINE
y = borrow << 1 NEW_LINE return x NEW_LINE
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
def subtract ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : NEW_LINE INDENT return x NEW_LINE DEDENT return subtract ( x ^ y , ( ~ x & y ) << 1 ) NEW_LINE DEDENT
x = 29 NEW_LINE y = 13 NEW_LINE print ( " x ▁ - ▁ y ▁ is " , subtract ( x , y ) ) NEW_LINE
N = 6 NEW_LINE Even = N // 2 NEW_LINE Odd = N - Even NEW_LINE print ( Even * Odd ) NEW_LINE
def steps ( string , n ) : NEW_LINE
flag = False NEW_LINE x = 0 NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE
if ( x == 0 ) : NEW_LINE INDENT flag = True NEW_LINE DEDENT
if ( x == n - 1 ) : NEW_LINE INDENT flag = False NEW_LINE DEDENT
for j in range ( x ) : NEW_LINE INDENT print ( " * " , end = " " ) NEW_LINE DEDENT print ( string [ i ] ) NEW_LINE
if ( flag == True ) : NEW_LINE INDENT x += 1 NEW_LINE DEDENT else : NEW_LINE INDENT x -= 1 NEW_LINE DEDENT
n = 4 NEW_LINE string = " GeeksForGeeks " NEW_LINE print ( " String : ▁ " , string ) NEW_LINE print ( " Max ▁ Length ▁ of ▁ Steps : ▁ " , n ) NEW_LINE
steps ( string , n ) NEW_LINE
def isDivisible ( str , k ) : NEW_LINE INDENT n = len ( str ) NEW_LINE c = 0 NEW_LINE DEDENT
for i in range ( 0 , k ) : NEW_LINE INDENT if ( str [ n - i - 1 ] == '0' ) : NEW_LINE INDENT c += 1 NEW_LINE DEDENT DEDENT
return ( c == k ) NEW_LINE
str1 = "10101100" NEW_LINE k = 2 NEW_LINE if ( isDivisible ( str1 , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
str2 = "111010100" NEW_LINE k = 2 NEW_LINE if ( isDivisible ( str2 , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def isNumber ( s ) : NEW_LINE INDENT for i in range ( len ( s ) ) : NEW_LINE INDENT if s [ i ] . isdigit ( ) != True : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE DEDENT
str = "6790" NEW_LINE
if isNumber ( str ) : NEW_LINE INDENT print ( " Integer " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( " String " ) NEW_LINE DEDENT
def reverse ( string ) : NEW_LINE INDENT if len ( string ) == 0 : NEW_LINE INDENT return NEW_LINE DEDENT temp = string [ 0 ] NEW_LINE reverse ( string [ 1 : ] ) NEW_LINE print ( temp , end = ' ' ) NEW_LINE DEDENT
string = " Geeks ▁ for ▁ Geeks " NEW_LINE reverse ( string ) NEW_LINE
def polyarea ( n , r ) : NEW_LINE
if ( r < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
A = ( ( ( r * r * n ) * sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ) ; NEW_LINE return round ( A , 3 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT r , n = 9 , 6 NEW_LINE print ( polyarea ( n , r ) ) NEW_LINE DEDENT
def findPCSlope ( m ) : NEW_LINE INDENT return - 1.0 / m NEW_LINE DEDENT
m = 2.0 NEW_LINE print ( findPCSlope ( m ) ) NEW_LINE
def area_of_segment ( radius , angle ) : NEW_LINE
area_of_sector = pi * NEW_LINE INDENT ( radius * radius ) NEW_LINE * ( angle / 360 ) NEW_LINE DEDENT
area_of_triangle = 1 / 2 * NEW_LINE INDENT ( radius * radius ) * NEW_LINE math . sin ( ( angle * pi ) / 180 ) NEW_LINE DEDENT return area_of_sector - area_of_triangle ; NEW_LINE
radius = 10.0 NEW_LINE angle = 90.0 NEW_LINE print ( " Area ▁ of ▁ minor ▁ segment ▁ = " , area_of_segment ( radius , angle ) ) NEW_LINE print ( " Area ▁ of ▁ major ▁ segment ▁ = " , area_of_segment ( radius , ( 360 - angle ) ) ) NEW_LINE
def SectorArea ( radius , angle ) : NEW_LINE INDENT pi = 22 / 7 NEW_LINE if angle >= 360 : NEW_LINE INDENT print ( " Angle ▁ not ▁ possible " ) NEW_LINE return NEW_LINE DEDENT DEDENT
else : NEW_LINE INDENT sector = ( pi * radius ** 2 ) * ( angle / 360 ) NEW_LINE print ( sector ) NEW_LINE return NEW_LINE DEDENT
radius = 9 NEW_LINE angle = 60 NEW_LINE SectorArea ( radius , angle ) NEW_LINE
def insertionSortRecursive ( arr , n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return NEW_LINE DEDENT
insertionSortRecursive ( arr , n - 1 ) NEW_LINE
last = arr [ n - 1 ] NEW_LINE j = n - 2 NEW_LINE
while ( j >= 0 and arr [ j ] > last ) : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j = j - 1 NEW_LINE DEDENT arr [ j + 1 ] = last NEW_LINE
def printArray ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT print arr [ i ] , NEW_LINE DEDENT DEDENT
arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE insertionSortRecursive ( arr , n ) NEW_LINE printArray ( arr , n ) NEW_LINE
def isWaveArray ( arr , n ) : NEW_LINE INDENT result = True NEW_LINE DEDENT
if ( arr [ 1 ] > arr [ 0 ] and arr [ 1 ] > arr [ 2 ] ) : NEW_LINE INDENT for i in range ( 1 , n - 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] > arr [ i - 1 ] and arr [ i ] > arr [ i + 1 ] ) : NEW_LINE INDENT result = True NEW_LINE DEDENT else : NEW_LINE INDENT result = False NEW_LINE break NEW_LINE DEDENT DEDENT DEDENT
if ( result == True and n % 2 == 0 ) : NEW_LINE INDENT if ( arr [ n - 1 ] <= arr [ n - 2 ] ) : NEW_LINE INDENT result = False NEW_LINE DEDENT DEDENT elif ( arr [ 1 ] < arr [ 0 ] and arr [ 1 ] < arr [ 2 ] ) : NEW_LINE for i in range ( 1 , n - 1 , 2 ) : NEW_LINE INDENT if ( arr [ i ] < arr [ i - 1 ] and arr [ i ] < arr [ i + 1 ] ) : NEW_LINE INDENT result = True NEW_LINE DEDENT else : NEW_LINE INDENT result = False NEW_LINE break NEW_LINE DEDENT DEDENT
if ( result == True and n % 2 == 0 ) : NEW_LINE INDENT if ( arr [ n - 1 ] >= arr [ n - 2 ] ) : NEW_LINE INDENT result = False NEW_LINE DEDENT DEDENT return result NEW_LINE
arr = [ 1 , 3 , 2 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE if ( isWaveArray ( arr , n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
mod = 1000000007 ; NEW_LINE
def sumOddFibonacci ( n ) : NEW_LINE INDENT Sum = [ 0 ] * ( n + 1 ) ; NEW_LINE DEDENT
Sum [ 0 ] = 0 ; NEW_LINE Sum [ 1 ] = 1 ; NEW_LINE Sum [ 2 ] = 2 ; NEW_LINE Sum [ 3 ] = 5 ; NEW_LINE Sum [ 4 ] = 10 ; NEW_LINE Sum [ 5 ] = 23 ; NEW_LINE for i in range ( 6 , n + 1 ) : NEW_LINE INDENT Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; NEW_LINE DEDENT return Sum [ n ] ; NEW_LINE
n = 6 ; NEW_LINE print ( sumOddFibonacci ( n ) ) ; NEW_LINE
def solve ( N , K ) : NEW_LINE
combo = [ 0 ] * ( N + 1 ) NEW_LINE
combo [ 0 ] = 1 NEW_LINE
for i in range ( 1 , K + 1 ) : NEW_LINE
for j in range ( 0 , N + 1 ) : NEW_LINE
if j >= i : NEW_LINE
combo [ j ] += combo [ j - i ] NEW_LINE
return combo [ N ] NEW_LINE
N , K = 29 , 5 NEW_LINE print ( solve ( N , K ) ) NEW_LINE
def computeLIS ( circBuff , start , end , n ) : NEW_LINE INDENT LIS = [ 0 for i in range ( end ) ] NEW_LINE DEDENT
for i in range ( start , end ) : NEW_LINE INDENT LIS [ i ] = 1 NEW_LINE DEDENT
for i in range ( start + 1 , end ) : NEW_LINE
for j in range ( start , i ) : NEW_LINE INDENT if ( circBuff [ i ] > circBuff [ j ] and LIS [ i ] < LIS [ j ] + 1 ) : NEW_LINE INDENT LIS [ i ] = LIS [ j ] + 1 NEW_LINE DEDENT DEDENT
res = - 100000 NEW_LINE for i in range ( start , end ) : NEW_LINE INDENT res = max ( res , LIS [ i ] ) NEW_LINE DEDENT return res NEW_LINE
def LICS ( arr , n ) : NEW_LINE
circBuff = [ 0 for i in range ( 2 * n ) ] NEW_LINE for i in range ( n ) : NEW_LINE INDENT circBuff [ i ] = arr [ i ] NEW_LINE DEDENT for i in range ( n , 2 * n ) : NEW_LINE INDENT circBuff [ i ] = arr [ i - n ] NEW_LINE DEDENT
res = - 100000 NEW_LINE for i in range ( n ) : NEW_LINE INDENT res = max ( computeLIS ( circBuff , i , i + n , n ) , res ) NEW_LINE DEDENT return res NEW_LINE
arr = [ 1 , 4 , 6 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Length ▁ of ▁ LICS ▁ is " , LICS ( arr , n ) ) NEW_LINE
def LCIS ( arr1 , n , arr2 , m ) : NEW_LINE
table = [ 0 ] * m NEW_LINE for j in range ( m ) : NEW_LINE INDENT table [ j ] = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
current = 0 NEW_LINE
for j in range ( m ) : NEW_LINE
if ( arr1 [ i ] == arr2 [ j ] ) : NEW_LINE INDENT if ( current + 1 > table [ j ] ) : NEW_LINE INDENT table [ j ] = current + 1 NEW_LINE DEDENT DEDENT
if ( arr1 [ i ] > arr2 [ j ] ) : NEW_LINE INDENT if ( table [ j ] > current ) : NEW_LINE INDENT current = table [ j ] NEW_LINE DEDENT DEDENT
result = 0 NEW_LINE for i in range ( m ) : NEW_LINE INDENT if ( table [ i ] > result ) : NEW_LINE INDENT result = table [ i ] NEW_LINE DEDENT DEDENT return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr1 = [ 3 , 4 , 9 , 1 ] NEW_LINE arr2 = [ 5 , 3 , 8 , 9 , 10 , 2 , 1 ] NEW_LINE n = len ( arr1 ) NEW_LINE m = len ( arr2 ) NEW_LINE print ( " Length ▁ of ▁ LCIS ▁ is " , LCIS ( arr1 , n , arr2 , m ) ) NEW_LINE DEDENT
def maxValue ( a , b ) : NEW_LINE
b = sorted ( b ) NEW_LINE bi = [ i for i in b ] NEW_LINE ai = [ i for i in a ] NEW_LINE n = len ( a ) NEW_LINE m = len ( b ) NEW_LINE
j = m - 1 NEW_LINE for i in range ( n ) : NEW_LINE
if ( j < 0 ) : NEW_LINE INDENT break NEW_LINE DEDENT if ( bi [ j ] > ai [ i ] ) : NEW_LINE INDENT ai [ i ] = bi [ j ] NEW_LINE DEDENT
j -= 1 NEW_LINE
x = " " . join ( ai ) NEW_LINE return x NEW_LINE
a = "1234" NEW_LINE b = "4321" NEW_LINE print ( maxValue ( a , b ) ) NEW_LINE
def checkIfUnequal ( n , q ) : NEW_LINE
s1 = str ( n ) NEW_LINE a = [ 0 for i in range ( 26 ) ] NEW_LINE
for i in range ( 0 , len ( s1 ) , 1 ) : NEW_LINE INDENT a [ ord ( s1 [ i ] ) - ord ( '0' ) ] += 1 NEW_LINE DEDENT
prod = n * q NEW_LINE
s2 = str ( prod ) NEW_LINE
for i in range ( 0 , len ( s2 ) , 1 ) : NEW_LINE
if ( a [ ord ( s2 [ i ] ) - ord ( '0' ) ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return True NEW_LINE
def countInRange ( l , r , q ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( l , r + 1 , 1 ) : NEW_LINE DEDENT
if ( checkIfUnequal ( i , q ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT l = 10 NEW_LINE r = 12 NEW_LINE q = 2 NEW_LINE DEDENT
print ( countInRange ( l , r , q ) ) NEW_LINE
def is_possible ( s ) : NEW_LINE
l = len ( s ) NEW_LINE one = 0 NEW_LINE zero = 0 NEW_LINE for i in range ( 0 , l ) : NEW_LINE
if ( s [ i ] == '0' ) : NEW_LINE INDENT zero += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT one += 1 NEW_LINE DEDENT
if ( l % 2 == 0 ) : NEW_LINE INDENT return ( one == zero ) NEW_LINE DEDENT
else : NEW_LINE INDENT return ( abs ( one - zero ) == 1 ) NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "100110" NEW_LINE if ( is_possible ( s ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def convert ( s ) : NEW_LINE INDENT n = len ( s ) NEW_LINE s1 = " " NEW_LINE s1 = s1 + s [ 0 ] . lower ( ) NEW_LINE i = 1 NEW_LINE while i < n : NEW_LINE DEDENT
if ( s [ i ] == ' ▁ ' and i <= n ) : NEW_LINE
s1 = s1 + " ▁ " + ( s [ i + 1 ] ) . lower ( ) NEW_LINE i = i + 1 NEW_LINE
else : NEW_LINE INDENT s1 = s1 + ( s [ i ] ) . upper ( ) NEW_LINE DEDENT i = i + 1 NEW_LINE
return s1 NEW_LINE
str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " NEW_LINE print ( convert ( str ) ) NEW_LINE
def reverse ( s ) : NEW_LINE INDENT if len ( s ) == 0 : NEW_LINE INDENT return s NEW_LINE DEDENT else : NEW_LINE INDENT return reverse ( s [ 1 : ] ) + s [ 0 ] NEW_LINE DEDENT DEDENT def findNthNo ( n ) : NEW_LINE INDENT res = " " ; NEW_LINE while ( n >= 1 ) : NEW_LINE DEDENT
if ( n & 1 ) : NEW_LINE INDENT res = res + "3" ; NEW_LINE n = ( int ) ( ( n - 1 ) / 2 ) ; NEW_LINE DEDENT
else : NEW_LINE INDENT res = res + "5" ; NEW_LINE n = ( int ) ( ( n - 2 ) / 2 ) ; NEW_LINE DEDENT
return reverse ( res ) ; NEW_LINE
n = 5 ; NEW_LINE print ( findNthNo ( n ) ) ; NEW_LINE
def findNthNonSquare ( n ) : NEW_LINE
x = n ; NEW_LINE
ans = x + math . floor ( 0.5 + math . sqrt ( x ) ) ; NEW_LINE return int ( ans ) ; NEW_LINE
n = 16 ; NEW_LINE
print ( " The " , n , " th ▁ Non - Square ▁ number ▁ is " , findNthNonSquare ( n ) ) ; NEW_LINE
def seiresSum ( n , a ) : NEW_LINE INDENT return ( n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ) NEW_LINE DEDENT
n = 2 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE print ( int ( seiresSum ( n , a ) ) ) NEW_LINE
def checkdigit ( n , k ) : NEW_LINE INDENT while ( n ) : NEW_LINE DEDENT
rem = n % 10 NEW_LINE
if ( rem == k ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT n = n / 10 NEW_LINE return 0 NEW_LINE
def findNthNumber ( n , k ) : NEW_LINE
i = k + 1 NEW_LINE count = 1 NEW_LINE while ( count < n ) : NEW_LINE
if ( checkdigit ( i , k ) or ( i % k == 0 ) ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT if ( count == n ) : NEW_LINE INDENT return i NEW_LINE DEDENT i += 1 NEW_LINE return - 1 NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( findNthNumber ( n , k ) ) NEW_LINE
def middleOfThree ( a , b , c ) : NEW_LINE
INDENT def middleOfThree ( a , b , c ) : NEW_LINE DEDENT
if ( ( a < b and b < c ) or ( c < b and b < a ) ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT
if ( ( b < a and a < c ) or ( c < a and a < b ) ) : NEW_LINE INDENT return a ; NEW_LINE DEDENT else : NEW_LINE INDENT return c NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
INF = 2147483647 NEW_LINE N = 4 NEW_LINE
def minCost ( cost ) : NEW_LINE
dist = [ 0 for i in range ( N ) ] NEW_LINE for i in range ( N ) : NEW_LINE INDENT dist [ i ] = INF NEW_LINE DEDENT dist [ 0 ] = 0 NEW_LINE
for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE INDENT if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) : NEW_LINE INDENT dist [ j ] = dist [ i ] + cost [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT return dist [ N - 1 ] NEW_LINE
cost = [ [ 0 , 15 , 80 , 90 ] , [ INF , 0 , 40 , 50 ] , [ INF , INF , 0 , 70 ] , [ INF , INF , INF , 0 ] ] NEW_LINE print ( " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " , N , " ▁ is ▁ " , minCost ( cost ) ) NEW_LINE
def numOfways ( n , k ) : NEW_LINE INDENT p = 1 NEW_LINE if ( k % 2 ) : NEW_LINE INDENT p = - 1 NEW_LINE DEDENT return ( pow ( n - 1 , k ) + p * ( n - 1 ) ) / n NEW_LINE DEDENT
n = 4 NEW_LINE k = 2 NEW_LINE print ( numOfways ( n , k ) ) NEW_LINE
def length_of_chord ( r , x ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ chord " , " ▁ of ▁ the ▁ circle ▁ is ▁ " , 2 * r * mt . sin ( x * ( 3.14 / 180 ) ) ) NEW_LINE DEDENT
r = 4 NEW_LINE x = 63 ; NEW_LINE length_of_chord ( r , x ) NEW_LINE
def area ( a ) : NEW_LINE
if a < 0 : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
area = sqrt ( a ) / 6 NEW_LINE return area NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 10 NEW_LINE print ( round ( area ( a ) , 6 ) ) NEW_LINE DEDENT
def longestRodInCuboid ( length , breadth , height ) : NEW_LINE
temp = length * length + breadth * breadth + height * height NEW_LINE
result = sqrt ( temp ) NEW_LINE return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT length , breadth , height = 12 , 9 , 8 NEW_LINE DEDENT
print ( longestRodInCuboid ( length , breadth , height ) ) NEW_LINE
def LiesInsieRectangle ( a , b , x , y ) : NEW_LINE INDENT if ( x - y - b <= 0 and x - y + b >= 0 and x + y - 2 * a + b <= 0 and x + y - b >= 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a , b , x , y = 7 , 2 , 4 , 5 NEW_LINE if LiesInsieRectangle ( a , b , x , y ) : NEW_LINE INDENT print ( " Given ▁ point ▁ lies ▁ inside " " ▁ the ▁ rectangle " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Given ▁ point ▁ does ▁ not ▁ lie " " ▁ on ▁ the ▁ rectangle " ) NEW_LINE DEDENT DEDENT
def maxvolume ( s ) : NEW_LINE INDENT maxvalue = 0 NEW_LINE DEDENT
i = 1 NEW_LINE for i in range ( s - 1 ) : NEW_LINE INDENT j = 1 NEW_LINE DEDENT
for j in range ( s ) : NEW_LINE
k = s - i - j NEW_LINE
maxvalue = max ( maxvalue , i * j * k ) NEW_LINE return maxvalue NEW_LINE
s = 8 NEW_LINE print ( maxvolume ( s ) ) NEW_LINE
def maxvolume ( s ) : NEW_LINE
length = int ( s / 3 ) NEW_LINE s -= length NEW_LINE
breadth = s / 2 NEW_LINE
height = s - breadth NEW_LINE return int ( length * breadth * height ) NEW_LINE
s = 8 NEW_LINE print ( maxvolume ( s ) ) NEW_LINE
def hexagonArea ( s ) : NEW_LINE INDENT return ( ( 3 * math . sqrt ( 3 ) * ( s * s ) ) / 2 ) ; NEW_LINE DEDENT
s = 4 NEW_LINE print ( " Area : " , " { 0 : . 4f } " . format ( hexagonArea ( s ) ) ) NEW_LINE
def maxSquare ( b , m ) : NEW_LINE
return ( b / m - 1 ) * ( b / m ) / 2 NEW_LINE
b = 10 NEW_LINE m = 2 NEW_LINE print ( int ( maxSquare ( b , m ) ) ) NEW_LINE
def findRightAngle ( A , H ) : NEW_LINE
D = pow ( H , 4 ) - 16 * A * A NEW_LINE if D >= 0 : NEW_LINE
root1 = ( H * H + sqrt ( D ) ) / 2 NEW_LINE root2 = ( H * H - sqrt ( D ) ) / 2 NEW_LINE a = sqrt ( root1 ) NEW_LINE b = sqrt ( root2 ) NEW_LINE if b >= a : NEW_LINE INDENT print a , b , H NEW_LINE DEDENT else : NEW_LINE INDENT print b , a , H NEW_LINE DEDENT else : NEW_LINE print " - 1" NEW_LINE
findRightAngle ( 6 , 5 ) NEW_LINE
def numberOfSquares ( base ) : NEW_LINE
base = ( base - 2 ) NEW_LINE
base = base // 2 NEW_LINE return base * ( base + 1 ) / 2 NEW_LINE
base = 8 NEW_LINE print ( numberOfSquares ( base ) ) NEW_LINE
def fib ( n ) : NEW_LINE INDENT if n <= 1 : NEW_LINE INDENT return n NEW_LINE DEDENT return fib ( n - 1 ) + fib ( n - 2 ) NEW_LINE DEDENT
def findVertices ( n ) : NEW_LINE
return fib ( n + 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 3 NEW_LINE print ( findVertices ( n ) ) NEW_LINE DEDENT
MAX_SIZE = 10 NEW_LINE
def sortByRow ( mat , n , ascending ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if ( ascending ) : NEW_LINE INDENT mat [ i ] . sort ( ) NEW_LINE DEDENT else : NEW_LINE INDENT mat [ i ] . sort ( reverse = True ) NEW_LINE DEDENT DEDENT DEDENT
def transpose ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT DEDENT
temp = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE mat [ j ] [ i ] = temp NEW_LINE
def sortMatRowAndColWise ( mat , n ) : NEW_LINE
sortByRow ( mat , n , True ) NEW_LINE
transpose ( mat , n ) NEW_LINE
sortByRow ( mat , n , False ) NEW_LINE
transpose ( mat , n ) NEW_LINE
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( mat [ i ] [ j ] , " ▁ " , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT DEDENT
n = 3 NEW_LINE mat = [ [ 3 , 2 , 1 ] , [ 9 , 8 , 7 ] , [ 6 , 5 , 4 ] ] NEW_LINE print ( " Original ▁ Matrix : " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMatRowAndColWise ( mat , n ) NEW_LINE print ( " Matrix ▁ After ▁ Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE
MAX_SIZE = 10 NEW_LINE
def sortByRow ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE DEDENT
for j in range ( n - 1 ) : NEW_LINE INDENT if mat [ i ] [ j ] > mat [ i ] [ j + 1 ] : NEW_LINE INDENT temp = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ i ] [ j + 1 ] NEW_LINE mat [ i ] [ j + 1 ] = temp NEW_LINE DEDENT DEDENT
def transpose ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( i + 1 , n ) : NEW_LINE DEDENT DEDENT
t = mat [ i ] [ j ] NEW_LINE mat [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE mat [ j ] [ i ] = t NEW_LINE
def sortMatRowAndColWise ( mat , n ) : NEW_LINE
sortByRow ( mat , n ) NEW_LINE
transpose ( mat , n ) NEW_LINE
sortByRow ( mat , n ) NEW_LINE
transpose ( mat , n ) NEW_LINE
def printMat ( mat , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ( str ( mat [ i ] [ j ] ) , end = " ▁ " ) NEW_LINE DEDENT print ( ) ; NEW_LINE DEDENT DEDENT
mat = [ [ 4 , 1 , 3 ] , [ 9 , 6 , 8 ] , [ 5 , 2 , 7 ] ] NEW_LINE n = 3 NEW_LINE print ( " Original ▁ Matrix : " ) NEW_LINE printMat ( mat , n ) NEW_LINE sortMatRowAndColWise ( mat , n ) NEW_LINE print ( " Matrix After Sorting : " ) NEW_LINE printMat ( mat , n ) NEW_LINE
def DoublyEven ( n ) : NEW_LINE
arr = [ [ ( n * y ) + x + 1 for x in range ( n ) ] for y in range ( n ) ] NEW_LINE
for i in range ( 0 , n / 4 ) : NEW_LINE INDENT for j in range ( 0 , n / 4 ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 0 , n / 4 ) : NEW_LINE INDENT for j in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT for j in range ( 0 , n / 4 ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT for j in range ( 3 * ( n / 4 ) , n ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( n / 4 , 3 * ( n / 4 ) ) : NEW_LINE INDENT for j in range ( n / 4 , 3 * ( n / 4 ) ) : NEW_LINE INDENT arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ; NEW_LINE DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT print ' % 2d ▁ ' % ( arr [ i ] [ j ] ) , NEW_LINE DEDENT print NEW_LINE DEDENT
n = 8 NEW_LINE
DoublyEven ( n ) NEW_LINE
cola = 2 NEW_LINE rowa = 3 NEW_LINE colb = 3 NEW_LINE rowb = 2 NEW_LINE
def Kroneckerproduct ( A , B ) : NEW_LINE INDENT C = [ [ 0 for j in range ( cola * colb ) ] for i in range ( rowa * rowb ) ] NEW_LINE DEDENT
for i in range ( 0 , rowa ) : NEW_LINE
for k in range ( 0 , rowb ) : NEW_LINE
for j in range ( 0 , cola ) : NEW_LINE
for l in range ( 0 , colb ) : NEW_LINE
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] NEW_LINE print ( C [ i + l + 1 ] [ j + k + 1 ] , end = ' ▁ ' ) NEW_LINE print ( " " ) NEW_LINE
A = [ [ 0 for j in range ( 2 ) ] for i in range ( 3 ) ] NEW_LINE B = [ [ 0 for j in range ( 3 ) ] for i in range ( 2 ) ] NEW_LINE A [ 0 ] [ 0 ] = 1 NEW_LINE A [ 0 ] [ 1 ] = 2 NEW_LINE A [ 1 ] [ 0 ] = 3 NEW_LINE A [ 1 ] [ 1 ] = 4 NEW_LINE A [ 2 ] [ 0 ] = 1 NEW_LINE A [ 2 ] [ 1 ] = 0 NEW_LINE B [ 0 ] [ 0 ] = 0 NEW_LINE B [ 0 ] [ 1 ] = 5 NEW_LINE B [ 0 ] [ 2 ] = 2 NEW_LINE B [ 1 ] [ 0 ] = 6 NEW_LINE B [ 1 ] [ 1 ] = 7 NEW_LINE B [ 1 ] [ 2 ] = 3 NEW_LINE Kroneckerproduct ( A , B ) NEW_LINE
def islowertriangular ( M ) : NEW_LINE INDENT for i in range ( 0 , len ( M ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( M ) ) : NEW_LINE INDENT if ( M [ i ] [ j ] != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
M = [ [ 1 , 0 , 0 , 0 ] , [ 1 , 4 , 0 , 0 ] , [ 4 , 6 , 2 , 0 ] , [ 0 , 4 , 7 , 6 ] ] NEW_LINE
if islowertriangular ( M ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def isuppertriangular ( M ) : NEW_LINE INDENT for i in range ( 1 , len ( M ) ) : NEW_LINE INDENT for j in range ( 0 , i ) : NEW_LINE INDENT if ( M [ i ] [ j ] != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
M = [ [ 1 , 3 , 5 , 3 ] , [ 0 , 4 , 6 , 2 ] , [ 0 , 0 , 2 , 5 ] , [ 0 , 0 , 0 , 6 ] ] NEW_LINE if isuppertriangular ( M ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
m = 3 NEW_LINE
n = 2 NEW_LINE
def countSets ( a ) : NEW_LINE
res = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT u = 0 NEW_LINE v = 0 NEW_LINE for j in range ( m ) : NEW_LINE INDENT if a [ i ] [ j ] : NEW_LINE INDENT u += 1 NEW_LINE DEDENT else : NEW_LINE INDENT v += 1 NEW_LINE DEDENT DEDENT res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 NEW_LINE DEDENT
for i in range ( m ) : NEW_LINE INDENT u = 0 NEW_LINE v = 0 NEW_LINE for j in range ( n ) : NEW_LINE INDENT if a [ j ] [ i ] : NEW_LINE INDENT u += 1 NEW_LINE DEDENT else : NEW_LINE INDENT v += 1 NEW_LINE DEDENT DEDENT res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 NEW_LINE DEDENT
return res - ( n * m ) NEW_LINE
a = [ [ 1 , 0 , 1 ] , [ 0 , 1 , 0 ] ] NEW_LINE print ( countSets ( a ) ) NEW_LINE
def transpose ( mat , tr , N ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT tr [ i ] [ j ] = mat [ j ] [ i ] NEW_LINE DEDENT DEDENT DEDENT
def isSymmetric ( mat , N ) : NEW_LINE INDENT tr = [ [ 0 for j in range ( len ( mat [ 0 ] ) ) ] for i in range ( len ( mat ) ) ] NEW_LINE transpose ( mat , tr , N ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] NEW_LINE if ( isSymmetric ( mat , 3 ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT
def isSymmetric ( mat , N ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT DEDENT return True NEW_LINE DEDENT
mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] NEW_LINE if ( isSymmetric ( mat , 3 ) ) : NEW_LINE INDENT print " Yes " NEW_LINE DEDENT else : NEW_LINE INDENT print " No " NEW_LINE DEDENT
MAX = 100 ; NEW_LINE
def findNormal ( mat , n ) : NEW_LINE INDENT sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( n ) : NEW_LINE INDENT sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; NEW_LINE DEDENT DEDENT return math . floor ( math . sqrt ( sum ) ) ; NEW_LINE DEDENT
def findTrace ( mat , n ) : NEW_LINE INDENT sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT sum += mat [ i ] [ i ] ; NEW_LINE DEDENT return sum ; NEW_LINE DEDENT
mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] ; NEW_LINE print ( " Trace ▁ of ▁ Matrix ▁ = " , findTrace ( mat , 5 ) ) ; NEW_LINE print ( " Normal ▁ of ▁ Matrix ▁ = " , findNormal ( mat , 5 ) ) ; NEW_LINE
def maxDet ( n ) : NEW_LINE INDENT return 2 * n * n * n NEW_LINE DEDENT
def resMatrix ( n ) : NEW_LINE INDENT for i in range ( 3 ) : NEW_LINE INDENT for j in range ( 3 ) : NEW_LINE DEDENT DEDENT
if i == 0 and j == 2 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT elif i == 1 and j == 0 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT elif i == 2 and j == 1 : NEW_LINE INDENT print ( "0" , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT print ( n , end = " ▁ " ) NEW_LINE DEDENT print ( " " ) NEW_LINE
n = 15 NEW_LINE print ( " Maximum ▁ Detrminat = " , maxDet ( n ) ) NEW_LINE print ( " Resultant ▁ Matrix : " ) NEW_LINE resMatrix ( n ) NEW_LINE
def countNegative ( M , n , m ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( m ) : NEW_LINE INDENT if M [ i ] [ j ] < 0 : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE DEDENT DEDENT
break NEW_LINE return count NEW_LINE
M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] NEW_LINE print ( countNegative ( M , 3 , 4 ) ) NEW_LINE
def countNegative ( M , n , m ) : NEW_LINE
count = 0 NEW_LINE
i = 0 NEW_LINE j = m - 1 NEW_LINE
while j >= 0 and i < n : NEW_LINE INDENT if M [ i ] [ j ] < 0 : NEW_LINE DEDENT
count += ( j + 1 ) NEW_LINE
i += 1 NEW_LINE else : NEW_LINE
j -= 1 NEW_LINE return count NEW_LINE
M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] NEW_LINE print ( countNegative ( M , 3 , 4 ) ) NEW_LINE
N = 5 NEW_LINE
def findMaxValue ( mat ) : NEW_LINE
maxValue = 0 NEW_LINE
for a in range ( N - 1 ) : NEW_LINE INDENT for b in range ( N - 1 ) : NEW_LINE INDENT for d in range ( a + 1 , N ) : NEW_LINE INDENT for e in range ( b + 1 , N ) : NEW_LINE INDENT if maxValue < int ( mat [ d ] [ e ] - mat [ a ] [ b ] ) : NEW_LINE INDENT maxValue = int ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ; NEW_LINE DEDENT DEDENT DEDENT DEDENT DEDENT return maxValue ; NEW_LINE
mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] ; NEW_LINE print ( " Maximum ▁ Value ▁ is ▁ " + str ( findMaxValue ( mat ) ) ) NEW_LINE
import sys NEW_LINE N = 5 NEW_LINE
def findMaxValue ( mat ) : NEW_LINE
maxValue = - sys . maxsize - 1 NEW_LINE
maxArr = [ [ 0 for x in range ( N ) ] for y in range ( N ) ] NEW_LINE
maxArr [ N - 1 ] [ N - 1 ] = mat [ N - 1 ] [ N - 1 ] NEW_LINE
maxv = mat [ N - 1 ] [ N - 1 ] ; NEW_LINE for j in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT if ( mat [ N - 1 ] [ j ] > maxv ) : NEW_LINE INDENT maxv = mat [ N - 1 ] [ j ] NEW_LINE DEDENT maxArr [ N - 1 ] [ j ] = maxv NEW_LINE DEDENT
maxv = mat [ N - 1 ] [ N - 1 ] ; NEW_LINE for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT if ( mat [ i ] [ N - 1 ] > maxv ) : NEW_LINE INDENT maxv = mat [ i ] [ N - 1 ] NEW_LINE DEDENT maxArr [ i ] [ N - 1 ] = maxv NEW_LINE DEDENT
for i in range ( N - 2 , - 1 , - 1 ) : NEW_LINE INDENT for j in range ( N - 2 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] > maxValue ) : NEW_LINE INDENT maxValue = ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] ) NEW_LINE DEDENT
maxArr [ i ] [ j ] = max ( mat [ i ] [ j ] , max ( maxArr [ i ] [ j + 1 ] , maxArr [ i + 1 ] [ j ] ) ) NEW_LINE return maxValue NEW_LINE
mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] NEW_LINE print ( " Maximum ▁ Value ▁ is " , findMaxValue ( mat ) ) NEW_LINE
n = 5 NEW_LINE
def printSumSimple ( mat , k ) : NEW_LINE
if ( k > n ) : NEW_LINE INDENT return NEW_LINE DEDENT
for i in range ( n - k + 1 ) : NEW_LINE
for j in range ( n - k + 1 ) : NEW_LINE
sum = 0 NEW_LINE for p in range ( i , k + i ) : NEW_LINE INDENT for q in range ( j , k + j ) : NEW_LINE INDENT sum += mat [ p ] [ q ] NEW_LINE DEDENT DEDENT print ( sum , end = " ▁ " ) NEW_LINE
print ( ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] NEW_LINE k = 3 NEW_LINE printSumSimple ( mat , k ) NEW_LINE DEDENT
n = 5 NEW_LINE
def printSumTricky ( mat , k ) : NEW_LINE INDENT global n NEW_LINE DEDENT
if k > n : NEW_LINE INDENT return NEW_LINE DEDENT
stripSum = [ [ None ] * n for i in range ( n ) ] NEW_LINE
for j in range ( n ) : NEW_LINE
Sum = 0 NEW_LINE for i in range ( k ) : NEW_LINE INDENT Sum += mat [ i ] [ j ] NEW_LINE DEDENT stripSum [ 0 ] [ j ] = Sum NEW_LINE
for i in range ( 1 , n - k + 1 ) : NEW_LINE INDENT Sum += ( mat [ i + k - 1 ] [ j ] - mat [ i - 1 ] [ j ] ) NEW_LINE stripSum [ i ] [ j ] = Sum NEW_LINE DEDENT
for i in range ( n - k + 1 ) : NEW_LINE
Sum = 0 NEW_LINE for j in range ( k ) : NEW_LINE INDENT Sum += stripSum [ i ] [ j ] NEW_LINE DEDENT print ( Sum , end = " ▁ " ) NEW_LINE
for j in range ( 1 , n - k + 1 ) : NEW_LINE INDENT Sum += ( stripSum [ i ] [ j + k - 1 ] - stripSum [ i ] [ j - 1 ] ) NEW_LINE print ( Sum , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE
mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] NEW_LINE k = 3 NEW_LINE printSumTricky ( mat , k ) NEW_LINE
M = 3 NEW_LINE N = 4 NEW_LINE
def transpose ( A , B ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( M ) : NEW_LINE INDENT B [ i ] [ j ] = A [ j ] [ i ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] ] NEW_LINE B = [ [ 0 for x in range ( M ) ] for y in range ( N ) ] NEW_LINE transpose ( A , B ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( M ) : NEW_LINE INDENT print ( B [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
N = 4 NEW_LINE
def transpose ( A ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( i + 1 , N ) : NEW_LINE INDENT A [ i ] [ j ] , A [ j ] [ i ] = A [ j ] [ i ] , A [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE transpose ( A ) NEW_LINE print ( " Modified ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( A [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
R = 3 NEW_LINE C = 3 NEW_LINE
def pathCountRec ( mat , m , n , k ) : NEW_LINE
if m < 0 or n < 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT elif m == 0 and n == 0 : NEW_LINE INDENT return k == mat [ m ] [ n ] NEW_LINE DEDENT
return ( pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ) NEW_LINE
def pathCount ( mat , k ) : NEW_LINE INDENT return pathCountRec ( mat , R - 1 , C - 1 , k ) NEW_LINE DEDENT
k = 12 NEW_LINE mat = [ [ 1 , 2 , 3 ] , [ 4 , 6 , 5 ] , [ 3 , 2 , 1 ] ] NEW_LINE print ( pathCount ( mat , k ) ) NEW_LINE
def bubbleSort ( arr ) : NEW_LINE INDENT n = len ( arr ) NEW_LINE for i in range ( n ) : NEW_LINE INDENT swapped = False NEW_LINE for j in range ( 0 , n - i - 1 ) : NEW_LINE INDENT if arr [ j ] > arr [ j + 1 ] : NEW_LINE DEDENT DEDENT DEDENT
arr [ j ] , arr [ j + 1 ] = arr [ j + 1 ] , arr [ j ] NEW_LINE swapped = True NEW_LINE
if swapped == False : NEW_LINE INDENT break NEW_LINE DEDENT
arr = [ 64 , 34 , 25 , 12 , 22 , 11 , 90 ] NEW_LINE bubbleSort ( arr ) NEW_LINE print ( " Sorted ▁ array ▁ : " ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( " % d " % arr [ i ] , end = " ▁ " ) NEW_LINE DEDENT
def findCrossOver ( arr , low , high , x ) : NEW_LINE
if ( arr [ high ] <= x ) : NEW_LINE INDENT return high NEW_LINE DEDENT
if ( arr [ low ] > x ) : NEW_LINE INDENT return low NEW_LINE DEDENT
mid = ( low + high ) // 2 NEW_LINE
if ( arr [ mid ] <= x and arr [ mid + 1 ] > x ) : NEW_LINE INDENT return mid NEW_LINE DEDENT
if ( arr [ mid ] < x ) : NEW_LINE INDENT return findCrossOver ( arr , mid + 1 , high , x ) NEW_LINE DEDENT return findCrossOver ( arr , low , mid - 1 , x ) NEW_LINE
def printKclosest ( arr , x , k , n ) : NEW_LINE
l = findCrossOver ( arr , 0 , n - 1 , x ) NEW_LINE
r = l + 1 NEW_LINE
count = 0 NEW_LINE
if ( arr [ l ] == x ) : NEW_LINE INDENT l -= 1 NEW_LINE DEDENT
while ( l >= 0 and r < n and count < k ) : NEW_LINE INDENT if ( x - arr [ l ] < arr [ r ] - x ) : NEW_LINE INDENT print ( arr [ l ] , end = " ▁ " ) NEW_LINE l -= 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( arr [ r ] , end = " ▁ " ) NEW_LINE r += 1 NEW_LINE DEDENT count += 1 NEW_LINE DEDENT
while ( count < k and l >= 0 ) : NEW_LINE INDENT print ( arr [ l ] , end = " ▁ " ) NEW_LINE l -= 1 NEW_LINE count += 1 NEW_LINE DEDENT
while ( count < k and r < n ) : NEW_LINE INDENT print ( arr [ r ] , end = " ▁ " ) NEW_LINE r += 1 NEW_LINE count += 1 NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 ] NEW_LINE n = len ( arr ) NEW_LINE x = 35 NEW_LINE k = 4 NEW_LINE printKclosest ( arr , x , 4 , n ) NEW_LINE DEDENT
def count ( S , m , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( m <= 0 and n >= 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE m = len ( arr ) NEW_LINE print ( count ( arr , m , 4 ) ) NEW_LINE
def count ( S , m , n ) : NEW_LINE
table = [ 0 for k in range ( n + 1 ) ] NEW_LINE
table [ 0 ] = 1 NEW_LINE
for i in range ( 0 , m ) : NEW_LINE INDENT for j in range ( S [ i ] , n + 1 ) : NEW_LINE INDENT table [ j ] += table [ j - S [ i ] ] NEW_LINE DEDENT DEDENT return table [ n ] NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE m = len ( arr ) NEW_LINE n = 4 NEW_LINE x = count ( arr , m , n ) NEW_LINE print ( x ) NEW_LINE
def MatrixChainOrder ( p , n ) : NEW_LINE
m = [ [ 0 for x in range ( n ) ] for x in range ( n ) ] NEW_LINE
for i in range ( 1 , n ) : NEW_LINE INDENT m [ i ] [ i ] = 0 NEW_LINE DEDENT
for L in range ( 2 , n ) : NEW_LINE INDENT for i in range ( 1 , n - L + 1 ) : NEW_LINE INDENT j = i + L - 1 NEW_LINE m [ i ] [ j ] = sys . maxint NEW_LINE for k in range ( i , j ) : NEW_LINE DEDENT DEDENT
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] NEW_LINE if q < m [ i ] [ j ] : NEW_LINE INDENT m [ i ] [ j ] = q NEW_LINE DEDENT return m [ 1 ] [ n - 1 ] NEW_LINE
arr = [ 1 , 2 , 3 , 4 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + str ( MatrixChainOrder ( arr , size ) ) ) NEW_LINE
def cutRod ( price , n ) : NEW_LINE INDENT if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT max_val = - sys . maxsize - 1 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT max_val = max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) NEW_LINE DEDENT return max_val NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is " , cutRod ( arr , size ) ) NEW_LINE
def cutRod ( price , n ) : NEW_LINE INDENT val = [ 0 for x in range ( n + 1 ) ] NEW_LINE val [ 0 ] = 0 NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT max_val = INT_MIN NEW_LINE for j in range ( i ) : NEW_LINE INDENT max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) NEW_LINE DEDENT val [ i ] = max_val NEW_LINE DEDENT return val [ n ] NEW_LINE
arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + str ( cutRod ( arr , size ) ) ) NEW_LINE
def multiply ( x , y ) : NEW_LINE
if ( y == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( y > 0 ) : NEW_LINE INDENT return ( x + multiply ( x , y - 1 ) ) NEW_LINE DEDENT
if ( y < 0 ) : NEW_LINE INDENT return - multiply ( x , - y ) NEW_LINE DEDENT
print ( multiply ( 5 , - 11 ) ) NEW_LINE
def SieveOfEratosthenes ( n ) : NEW_LINE
prime = [ True for i in range ( n + 1 ) ] NEW_LINE p = 2 NEW_LINE while ( p * p <= n ) : NEW_LINE
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * p , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False NEW_LINE DEDENT p += 1 NEW_LINE
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if prime [ p ] : NEW_LINE INDENT print p , NEW_LINE DEDENT DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 30 NEW_LINE print " Following ▁ are ▁ the ▁ prime ▁ numbers ▁ smaller " , NEW_LINE print " than ▁ or ▁ equal ▁ to " , n NEW_LINE SieveOfEratosthenes ( n ) NEW_LINE DEDENT
def binomialCoeff ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT for i in range ( 0 , k ) : NEW_LINE INDENT res = res * ( n - i ) NEW_LINE res = res // ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE DEDENT
def printPascal ( n ) : NEW_LINE
for line in range ( 0 , n ) : NEW_LINE
for i in range ( 0 , line + 1 ) : NEW_LINE INDENT print ( binomialCoeff ( line , i ) , " ▁ " , end = " " ) NEW_LINE DEDENT print ( ) NEW_LINE
n = 7 NEW_LINE printPascal ( n ) NEW_LINE
def printPascal ( n : int ) : NEW_LINE
arr = [ [ 0 for x in range ( n ) ] for y in range ( n ) ] NEW_LINE
for line in range ( 0 , n ) : NEW_LINE
for i in range ( 0 , line + 1 ) : NEW_LINE
if ( i == 0 or i == line ) : NEW_LINE INDENT arr [ line ] [ i ] = 1 NEW_LINE print ( arr [ line ] [ i ] , end = " ▁ " ) NEW_LINE DEDENT
else : NEW_LINE INDENT arr [ line ] [ i ] = ( arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ) NEW_LINE print ( arr [ line ] [ i ] , end = " ▁ " ) NEW_LINE DEDENT print ( " " , ▁ end ▁ = ▁ " " ) NEW_LINE
n = 5 NEW_LINE printPascal ( n ) NEW_LINE
def printPascal ( n ) : NEW_LINE INDENT for line in range ( 1 , n + 1 ) : NEW_LINE DEDENT
C = 1 ; NEW_LINE for i in range ( 1 , line + 1 ) : NEW_LINE
print ( C , end = " ▁ " ) ; NEW_LINE C = int ( C * ( line - i ) / i ) ; NEW_LINE print ( " " ) ; NEW_LINE
n = 5 ; NEW_LINE printPascal ( n ) ; NEW_LINE
def Add ( x , y ) : NEW_LINE
while ( y != 0 ) : NEW_LINE
carry = x & y NEW_LINE
x = x ^ y NEW_LINE
y = carry << 1 NEW_LINE return x NEW_LINE
print ( Add ( 15 , 32 ) ) NEW_LINE
def getModulo ( n , d ) : NEW_LINE INDENT return ( n & ( d - 1 ) ) NEW_LINE DEDENT
n = 6 NEW_LINE
d = 4 NEW_LINE print ( n , " moduo " , d , " is " , getModulo ( n , d ) ) NEW_LINE
def countSetBits ( n ) : NEW_LINE INDENT count = 0 NEW_LINE while ( n ) : NEW_LINE INDENT count += n & 1 NEW_LINE n >>= 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
i = 9 NEW_LINE print ( countSetBits ( i ) ) NEW_LINE
def countSetBits ( n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT else : NEW_LINE INDENT return 1 + countSetBits ( n & ( n - 1 ) ) NEW_LINE DEDENT
n = 9 NEW_LINE
print ( countSetBits ( n ) ) NEW_LINE
print ( bin ( 4 ) . count ( '1' ) ) ; NEW_LINE print ( bin ( 15 ) . count ( '1' ) ) ; NEW_LINE
num_to_bits = [ 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 ] ; NEW_LINE
def countSetBitsRec ( num ) : NEW_LINE INDENT nibble = 0 ; NEW_LINE if ( 0 == num ) : NEW_LINE INDENT return num_to_bits [ 0 ] ; NEW_LINE DEDENT DEDENT
nibble = num & 0xf ; NEW_LINE
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; NEW_LINE
num = 31 ; NEW_LINE print ( countSetBitsRec ( num ) ) ; NEW_LINE
def getParity ( n ) : NEW_LINE INDENT parity = 0 NEW_LINE while n : NEW_LINE INDENT parity = ~ parity NEW_LINE n = n & ( n - 1 ) NEW_LINE DEDENT return parity NEW_LINE DEDENT
n = 7 NEW_LINE print ( " Parity ▁ of ▁ no ▁ " , n , " ▁ = ▁ " , ( " odd " if getParity ( n ) else " even " ) ) NEW_LINE
import math NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( math . ceil ( Log2 ( n ) ) == math . floor ( Log2 ( n ) ) ) ; NEW_LINE DEDENT
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def isPowerOfTwo ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT while ( n != 1 ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT n = n // 2 NEW_LINE DEDENT return True NEW_LINE DEDENT
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def maxRepeating ( arr , n , k ) : NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT arr [ arr [ i ] % k ] += k NEW_LINE DEDENT
max = arr [ 0 ] NEW_LINE result = 0 NEW_LINE for i in range ( 1 , n ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE result = i NEW_LINE DEDENT DEDENT
return result NEW_LINE
arr = [ 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 ] NEW_LINE n = len ( arr ) NEW_LINE k = 8 NEW_LINE print ( " The ▁ maximum ▁ repeating ▁ number ▁ is " , maxRepeating ( arr , n , k ) ) NEW_LINE
def fun ( x ) : NEW_LINE INDENT y = ( x // 4 ) * 4 NEW_LINE DEDENT
ans = 0 NEW_LINE for i in range ( y , x + 1 ) : NEW_LINE INDENT ans ^= i NEW_LINE DEDENT return ans NEW_LINE
def query ( x ) : NEW_LINE
if ( x == 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT k = ( x + 1 ) // 2 NEW_LINE
if x % 2 == 0 : NEW_LINE INDENT return ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( 2 * fun ( k ) ) NEW_LINE DEDENT def allQueries ( q , l , r ) : NEW_LINE for i in range ( q ) : NEW_LINE INDENT print ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) NEW_LINE DEDENT
q = 3 NEW_LINE l = [ 2 , 2 , 5 ] NEW_LINE r = [ 4 , 8 , 9 ] NEW_LINE allQueries ( q , l , r ) NEW_LINE
def findMinSwaps ( arr , n ) : NEW_LINE
noOfZeroes = [ 0 ] * n NEW_LINE count = 0 NEW_LINE
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] NEW_LINE for i in range ( n - 2 , - 1 , - 1 ) : NEW_LINE INDENT noOfZeroes [ i ] = noOfZeroes [ i + 1 ] NEW_LINE if ( arr [ i ] == 0 ) : NEW_LINE INDENT noOfZeroes [ i ] = noOfZeroes [ i ] + 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] == 1 ) : NEW_LINE INDENT count = count + noOfZeroes [ i ] NEW_LINE DEDENT DEDENT return count NEW_LINE
arr = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE print ( findMinSwaps ( arr , n ) ) NEW_LINE
def printTwoOdd ( arr , size ) : NEW_LINE
xor2 = arr [ 0 ] NEW_LINE
set_bit_no = 0 NEW_LINE n = size - 2 NEW_LINE x , y = 0 , 0 NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT xor2 = xor2 ^ arr [ i ] NEW_LINE DEDENT
set_bit_no = xor2 & ~ ( xor2 - 1 ) NEW_LINE
for i in range ( size ) : NEW_LINE
if ( arr [ i ] & set_bit_no ) : NEW_LINE INDENT x = x ^ arr [ i ] NEW_LINE DEDENT
else : NEW_LINE INDENT y = y ^ arr [ i ] NEW_LINE DEDENT print ( " The ▁ two ▁ ODD ▁ elements ▁ are " , x , " & " , y ) NEW_LINE
arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE printTwoOdd ( arr , arr_size ) NEW_LINE
def findPair ( arr , n ) : NEW_LINE INDENT size = len ( arr ) NEW_LINE DEDENT
i , j = 0 , 1 NEW_LINE
while i < size and j < size : NEW_LINE INDENT if i != j and arr [ j ] - arr [ i ] == n : NEW_LINE INDENT print " Pair ▁ found ▁ ( " , arr [ i ] , " , " , arr [ j ] , " ) " NEW_LINE return True NEW_LINE DEDENT elif arr [ j ] - arr [ i ] < n : NEW_LINE INDENT j += 1 NEW_LINE DEDENT else : NEW_LINE INDENT i += 1 NEW_LINE DEDENT DEDENT print " No ▁ pair ▁ found " NEW_LINE return False NEW_LINE
arr = [ 1 , 8 , 30 , 40 , 100 ] NEW_LINE n = 60 NEW_LINE findPair ( arr , n ) NEW_LINE
def checkIsAP ( arr , n ) : NEW_LINE INDENT if ( n == 1 ) : return True NEW_LINE DEDENT
arr . sort ( ) NEW_LINE
d = arr [ 1 ] - arr [ 0 ] NEW_LINE for i in range ( 2 , n ) : NEW_LINE INDENT if ( arr [ i ] - arr [ i - 1 ] != d ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
arr = [ 20 , 15 , 5 , 0 , 10 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Yes " ) if ( checkIsAP ( arr , n ) ) else print ( " No " ) NEW_LINE
def countPairs ( a , n ) : NEW_LINE
mn = + 2147483647 NEW_LINE mx = - 2147483648 NEW_LINE for i in range ( n ) : NEW_LINE INDENT mn = min ( mn , a [ i ] ) NEW_LINE mx = max ( mx , a [ i ] ) NEW_LINE DEDENT
c1 = 0 NEW_LINE
c2 = 0 NEW_LINE for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] == mn ) : NEW_LINE INDENT c1 += 1 NEW_LINE DEDENT if ( a [ i ] == mx ) : NEW_LINE INDENT c2 += 1 NEW_LINE DEDENT DEDENT
if ( mn == mx ) : NEW_LINE INDENT return n * ( n - 1 ) // 2 NEW_LINE DEDENT else : NEW_LINE INDENT return c1 * c2 NEW_LINE DEDENT
a = [ 3 , 2 , 1 , 1 , 3 ] NEW_LINE n = len ( a ) NEW_LINE print ( countPairs ( a , n ) ) NEW_LINE
def findNumbers ( arr , n ) : NEW_LINE
sumN = ( n * ( n + 1 ) ) / 2 ; NEW_LINE
sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; NEW_LINE
sum = 0 ; NEW_LINE sumSq = 0 ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sum = sum + arr [ i ] ; NEW_LINE sumSq = sumSq + ( math . pow ( arr [ i ] , 2 ) ) ; NEW_LINE DEDENT B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; NEW_LINE A = sum - sumN + B ; NEW_LINE print ( " A ▁ = ▁ " , int ( A ) ) ; NEW_LINE print ( " B ▁ = ▁ " , int ( B ) ) ; NEW_LINE
arr = [ 1 , 2 , 2 , 3 , 4 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE findNumbers ( arr , n ) ; NEW_LINE
def countLessThan ( arr , n , key ) : NEW_LINE INDENT l = 0 NEW_LINE r = n - 1 NEW_LINE index = - 1 NEW_LINE DEDENT
while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if ( arr [ m ] < key ) : NEW_LINE INDENT l = m + 1 NEW_LINE index = m NEW_LINE DEDENT else : NEW_LINE INDENT r = m - 1 NEW_LINE DEDENT DEDENT return ( index + 1 ) NEW_LINE
def countGreaterThan ( arr , n , key ) : NEW_LINE INDENT l = 0 NEW_LINE r = n - 1 NEW_LINE index = - 1 NEW_LINE DEDENT
while ( l <= r ) : NEW_LINE INDENT m = ( l + r ) // 2 NEW_LINE if ( arr [ m ] <= key ) : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT else : NEW_LINE INDENT r = m - 1 NEW_LINE index = m NEW_LINE DEDENT DEDENT if ( index == - 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT return ( n - index ) NEW_LINE
def countTriplets ( n , a , b , c ) : NEW_LINE
a . sort NEW_LINE b . sort ( ) NEW_LINE c . sort ( ) NEW_LINE count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT current = b [ i ] NEW_LINE a_index = - 1 NEW_LINE c_index = - 1 NEW_LINE DEDENT
low = countLessThan ( a , n , current ) NEW_LINE
high = countGreaterThan ( c , n , current ) NEW_LINE
count += ( low * high ) NEW_LINE return count NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 5 ] NEW_LINE b = [ 2 , 4 ] NEW_LINE c = [ 3 , 6 ] NEW_LINE size = len ( a ) NEW_LINE print ( countTriplets ( size , a , b , c ) ) NEW_LINE DEDENT
def middleOfThree ( a , b , c ) : NEW_LINE
x = a - b NEW_LINE
y = b - c NEW_LINE
z = a - c NEW_LINE
if x * y > 0 : NEW_LINE INDENT return b NEW_LINE DEDENT
elif ( x * z > 0 ) : NEW_LINE INDENT return NEW_LINE DEDENT else : NEW_LINE INDENT return a NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
def missing4 ( arr ) : NEW_LINE
helper = [ 0 ] * 4 NEW_LINE
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT temp = abs ( arr [ i ] ) NEW_LINE DEDENT
if ( temp <= len ( arr ) ) : NEW_LINE INDENT arr [ temp - 1 ] = arr [ temp - 1 ] * ( - 1 ) NEW_LINE DEDENT
elif ( temp > len ( arr ) ) : NEW_LINE INDENT if ( temp % len ( arr ) ) : NEW_LINE INDENT helper [ temp % len ( arr ) - 1 ] = - 1 NEW_LINE DEDENT else : NEW_LINE INDENT helper [ ( temp % len ( arr ) ) + len ( arr ) - 1 ] = - 1 NEW_LINE DEDENT DEDENT
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT if ( arr [ i ] > 0 ) : NEW_LINE INDENT print ( ( i + 1 ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT for i in range ( 0 , len ( helper ) ) : NEW_LINE INDENT if ( helper [ i ] >= 0 ) : NEW_LINE INDENT print ( ( len ( arr ) + i + 1 ) , end = " ▁ " ) NEW_LINE DEDENT DEDENT
arr = [ 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 ] NEW_LINE missing4 ( arr ) NEW_LINE
def minMovesToSort ( arr , n ) : NEW_LINE INDENT moves = 0 NEW_LINE mn = arr [ n - 1 ] NEW_LINE for i in range ( n - 1 , - 1 , - 1 ) : NEW_LINE DEDENT
if ( arr [ i ] > mn ) : NEW_LINE INDENT moves += arr [ i ] - mn NEW_LINE DEDENT
return moves NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 3 , 5 , 2 , 8 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minMovesToSort ( arr , n ) ) NEW_LINE DEDENT
def findOptimalPairs ( arr , N ) : NEW_LINE INDENT arr . sort ( reverse = False ) NEW_LINE DEDENT
i = 0 NEW_LINE j = N - 1 NEW_LINE while ( i <= j ) : NEW_LINE INDENT print ( " ( " , arr [ i ] , " , " , arr [ j ] , " ) " , end = " ▁ " ) NEW_LINE i += 1 NEW_LINE j -= 1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 9 , 6 , 5 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE findOptimalPairs ( arr , N ) NEW_LINE DEDENT
def minOperations ( arr , n ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
freq = [ 0 ] * 1000001 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 NEW_LINE DEDENT
maxi = max ( arr ) NEW_LINE for i in range ( 1 , maxi + 1 ) : NEW_LINE INDENT if freq [ i ] != 0 : NEW_LINE DEDENT
for j in range ( i * 2 , maxi + 1 , i ) : NEW_LINE
freq [ j ] = 0 NEW_LINE
result += 1 NEW_LINE return result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 4 , 2 , 4 , 4 , 4 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minOperations ( arr , n ) ) NEW_LINE DEDENT
def minGCD ( arr , n ) : NEW_LINE INDENT minGCD = 0 ; NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT minGCD = gcd ( minGCD , arr [ i ] ) ; NEW_LINE DEDENT return minGCD ; NEW_LINE
def minLCM ( arr , n ) : NEW_LINE INDENT minLCM = arr [ 0 ] ; NEW_LINE DEDENT
for i in range ( 1 , n ) : NEW_LINE INDENT minLCM = min ( minLCM , arr [ i ] ) ; NEW_LINE DEDENT return minLCM ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 66 , 14 , 521 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( " LCM ▁ = ▁ " , minLCM ( arr , n ) , " , ▁ GCD ▁ = " , minGCD ( arr , n ) ) ; NEW_LINE DEDENT
def formStringMinOperations ( ss ) : NEW_LINE
count = [ 0 ] * 3 ; NEW_LINE s = list ( ss ) ; NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT count [ ord ( s [ i ] ) - ord ( '0' ) ] += 1 ; NEW_LINE DEDENT
processed = [ 0 ] * 3 ; NEW_LINE
reqd = math . floor ( len ( s ) / 3 ) ; NEW_LINE for i in range ( len ( s ) ) : NEW_LINE
if ( count [ ord ( s [ i ] ) - ord ( '0' ) ] == reqd ) : NEW_LINE INDENT continue ; NEW_LINE DEDENT
if ( s [ i ] == '0' and count [ 0 ] > reqd and processed [ 0 ] >= reqd ) : NEW_LINE
if ( count [ 1 ] < reqd ) : NEW_LINE INDENT s [ i ] = '1' ; NEW_LINE count [ 1 ] += 1 ; NEW_LINE count [ 0 ] -= 1 ; NEW_LINE DEDENT
elif ( count [ 2 ] < reqd ) : NEW_LINE INDENT s [ i ] = '2' ; NEW_LINE count [ 2 ] += 1 ; NEW_LINE count [ 0 ] -= 1 ; NEW_LINE DEDENT
if ( s [ i ] == '1' and count [ 1 ] > reqd ) : NEW_LINE INDENT if ( count [ 0 ] < reqd ) : NEW_LINE INDENT s [ i ] = '0' ; NEW_LINE count [ 0 ] += 1 ; NEW_LINE count [ 1 ] -= 1 ; NEW_LINE DEDENT elif ( count [ 2 ] < reqd and processed [ 1 ] >= reqd ) : NEW_LINE INDENT s [ i ] = '2' ; NEW_LINE count [ 2 ] += 1 ; NEW_LINE count [ 1 ] -= 1 ; NEW_LINE DEDENT DEDENT
if ( s [ i ] == '2' and count [ 2 ] > reqd ) : NEW_LINE INDENT if ( count [ 0 ] < reqd ) : NEW_LINE INDENT s [ i ] = '0' ; NEW_LINE count [ 0 ] += 1 ; NEW_LINE count [ 2 ] -= 1 ; NEW_LINE DEDENT elif ( count [ 1 ] < reqd ) : NEW_LINE INDENT s [ i ] = '1' ; NEW_LINE count [ 1 ] += 1 ; NEW_LINE count [ 2 ] -= 1 ; NEW_LINE DEDENT DEDENT
processed [ ord ( s [ i ] ) - ord ( '0' ) ] += 1 ; NEW_LINE return ' ' . join ( s ) ; NEW_LINE
s = "011200" ; NEW_LINE print ( formStringMinOperations ( s ) ) ; NEW_LINE
def FindMaximumSum ( ind , kon , a , b , c , n , dp ) : NEW_LINE
if ind == n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if dp [ ind ] [ kon ] != - 1 : NEW_LINE INDENT return dp [ ind ] [ kon ] NEW_LINE DEDENT ans = - 10 ** 9 + 5 NEW_LINE
if kon == 0 : NEW_LINE INDENT ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) NEW_LINE DEDENT
elif kon == 1 : NEW_LINE INDENT ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) NEW_LINE DEDENT
elif kon == 2 : NEW_LINE INDENT ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) NEW_LINE ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) NEW_LINE DEDENT dp [ ind ] [ kon ] = ans NEW_LINE return ans NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 3 NEW_LINE a = [ 6 , 8 , 2 , 7 , 4 , 2 , 7 ] NEW_LINE b = [ 7 , 8 , 5 , 8 , 6 , 3 , 5 ] NEW_LINE c = [ 8 , 3 , 2 , 6 , 8 , 4 , 1 ] NEW_LINE n = len ( a ) NEW_LINE dp = [ [ - 1 for i in range ( N ) ] for j in range ( n ) ] NEW_LINE DEDENT
x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) NEW_LINE
y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) NEW_LINE
z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) NEW_LINE
print ( max ( x , y , z ) ) NEW_LINE
mod = 1000000007 ; NEW_LINE
def noOfBinaryStrings ( N , k ) : NEW_LINE INDENT dp = [ 0 ] * 100002 ; NEW_LINE for i in range ( 1 , K ) : NEW_LINE INDENT dp [ i ] = 1 ; NEW_LINE DEDENT dp [ k ] = 2 ; NEW_LINE for i in range ( k + 1 , N + 1 ) : NEW_LINE INDENT dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; NEW_LINE DEDENT return dp [ N ] ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT N = 4 ; NEW_LINE K = 2 ; NEW_LINE print ( noOfBinaryStrings ( N , K ) ) ; NEW_LINE DEDENT
def findWays ( p ) : NEW_LINE
dp = [ 0 ] * ( p + 1 ) NEW_LINE dp [ 1 ] = 1 NEW_LINE dp [ 2 ] = 2 NEW_LINE
for i in range ( 3 , p + 1 ) : NEW_LINE INDENT dp [ i ] = ( dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ) NEW_LINE DEDENT return dp [ p ] NEW_LINE
p = 3 NEW_LINE print ( findWays ( p ) ) NEW_LINE
def CountWays ( n ) : NEW_LINE
if n == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT if n == 1 : NEW_LINE INDENT return 1 NEW_LINE DEDENT if n == 2 : NEW_LINE INDENT return 1 + 1 NEW_LINE DEDENT
return CountWays ( n - 1 ) + CountWays ( n - 3 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 5 NEW_LINE print ( CountWays ( n ) ) NEW_LINE DEDENT
def maxSubArraySumRepeated ( a , n , k ) : NEW_LINE INDENT max_so_far = - 2147483648 NEW_LINE max_ending_here = 0 NEW_LINE for i in range ( n * k ) : NEW_LINE DEDENT
max_ending_here = max_ending_here + a [ i % n ] NEW_LINE if ( max_so_far < max_ending_here ) : NEW_LINE INDENT max_so_far = max_ending_here NEW_LINE DEDENT if ( max_ending_here < 0 ) : NEW_LINE INDENT max_ending_here = 0 NEW_LINE DEDENT return max_so_far NEW_LINE
a = [ 10 , 20 , - 30 , - 1 ] NEW_LINE n = len ( a ) NEW_LINE k = 3 NEW_LINE print ( " Maximum ▁ contiguous ▁ sum ▁ is ▁ " , maxSubArraySumRepeated ( a , n , k ) ) NEW_LINE
def longOddEvenIncSeq ( arr , n ) : NEW_LINE
lioes = list ( ) NEW_LINE
maxLen = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT lioes . append ( 1 ) NEW_LINE DEDENT
i = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT for j in range ( i ) : NEW_LINE INDENT if ( arr [ i ] > arr [ j ] and ( arr [ i ] + arr [ j ] ) % 2 != 0 and lioes [ i ] < lioes [ j ] + 1 ) : NEW_LINE INDENT lioes [ i ] = lioes [ j ] + 1 NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n ) : NEW_LINE INDENT if maxLen < lioes [ i ] : NEW_LINE INDENT maxLen = lioes [ i ] NEW_LINE DEDENT DEDENT
return maxLen NEW_LINE
arr = [ 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Longest ▁ Increasing ▁ Odd ▁ Even ▁ " + " Subsequence : ▁ " , longOddEvenIncSeq ( arr , n ) ) NEW_LINE
def MatrixChainOrder ( p , i , j ) : NEW_LINE INDENT if i == j : NEW_LINE INDENT return 0 NEW_LINE DEDENT _min = sys . maxsize NEW_LINE DEDENT
for k in range ( i , j ) : NEW_LINE INDENT count = ( MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) NEW_LINE if count < _min : NEW_LINE INDENT _min = count NEW_LINE DEDENT DEDENT
return _min NEW_LINE
arr = [ 1 , 2 , 3 , 4 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " , MatrixChainOrder ( arr , 1 , n - 1 ) ) NEW_LINE
def getCount ( a , b ) : NEW_LINE
if ( len ( b ) % len ( a ) != 0 ) : NEW_LINE INDENT return - 1 ; NEW_LINE DEDENT count = int ( len ( b ) / len ( a ) ) NEW_LINE
a = a * count NEW_LINE if ( a == b ) : NEW_LINE INDENT return count NEW_LINE DEDENT return - 1 ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = ' geeks ' NEW_LINE b = ' geeksgeeks ' NEW_LINE print ( getCount ( a , b ) ) NEW_LINE DEDENT
def countPattern ( s ) : NEW_LINE INDENT length = len ( s ) NEW_LINE oneSeen = False NEW_LINE DEDENT
for i in range ( length ) : NEW_LINE
if ( s [ i ] == '1' and oneSeen ) : NEW_LINE INDENT if ( s [ i - 1 ] == '0' ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
if ( s [ i ] == '1' and oneSeen == 0 ) : NEW_LINE INDENT oneSeen = True NEW_LINE DEDENT
if ( s [ i ] != '0' and s [ i ] != '1' ) : NEW_LINE INDENT oneSeen = False NEW_LINE DEDENT return count NEW_LINE
s = "100001abc101" NEW_LINE print countPattern ( s ) NEW_LINE
def minOperations ( s , t , n ) : NEW_LINE INDENT ct0 = 0 NEW_LINE ct1 = 0 NEW_LINE for i in range ( n ) : NEW_LINE DEDENT
if ( s [ i ] == t [ i ] ) : NEW_LINE INDENT continue NEW_LINE DEDENT
if ( s [ i ] == '0' ) : NEW_LINE INDENT ct0 += 1 NEW_LINE DEDENT
else : NEW_LINE INDENT ct1 += 1 NEW_LINE DEDENT return max ( ct0 , ct1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "010" NEW_LINE t = "101" NEW_LINE n = len ( s ) NEW_LINE print ( minOperations ( s , t , n ) ) NEW_LINE DEDENT
def decryptString ( str , n ) : NEW_LINE
i = 0 NEW_LINE jump = 1 NEW_LINE decryptedStr = " " NEW_LINE while ( i < n ) : NEW_LINE INDENT decryptedStr += str [ i ] ; NEW_LINE i += jump NEW_LINE DEDENT
jump += 1 NEW_LINE return decryptedStr NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " geeeeekkkksssss " NEW_LINE n = len ( str ) NEW_LINE print ( decryptString ( str , n ) ) NEW_LINE DEDENT
def bitToBeFlipped ( s ) : NEW_LINE
last = s [ len ( s ) - 1 ] NEW_LINE first = s [ 0 ] NEW_LINE
if ( last == first ) : NEW_LINE INDENT if ( last == '0' ) : NEW_LINE INDENT return '1' NEW_LINE DEDENT else : NEW_LINE INDENT return '0' NEW_LINE DEDENT DEDENT
elif ( last != first ) : NEW_LINE INDENT return last NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "1101011000" NEW_LINE print ( bitToBeFlipped ( s ) ) NEW_LINE DEDENT
def findSubSequence ( s , num ) : NEW_LINE
res = 0 NEW_LINE
i = 0 NEW_LINE while ( num ) : NEW_LINE
if ( num & 1 ) : NEW_LINE INDENT res += ord ( s [ i ] ) - ord ( '0' ) NEW_LINE DEDENT i += 1 NEW_LINE
num = num >> 1 NEW_LINE return res NEW_LINE
def combinedSum ( s ) : NEW_LINE
n = len ( s ) NEW_LINE
c_sum = 0 NEW_LINE
ran = ( 1 << n ) - 1 NEW_LINE
for i in range ( ran + 1 ) : NEW_LINE INDENT c_sum += findSubSequence ( s , i ) NEW_LINE DEDENT
return c_sum NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT s = "123" NEW_LINE print ( combinedSum ( s ) ) NEW_LINE DEDENT
def findSubsequence ( stri , k ) : NEW_LINE
a = [ 0 ] * MAX_CHAR ; NEW_LINE
for i in range ( len ( stri ) ) : NEW_LINE INDENT a [ ord ( stri [ i ] ) - ord ( ' a ' ) ] += 1 NEW_LINE DEDENT
for i in range ( len ( stri ) ) : NEW_LINE INDENT if a [ ord ( stri [ i ] ) - ord ( ' a ' ) ] >= k : NEW_LINE INDENT print ( stri [ i ] , end = ' ' ) NEW_LINE DEDENT DEDENT
k = 2 NEW_LINE findSubsequence ( " geeksforgeeks " , k ) NEW_LINE
def convert ( str ) : NEW_LINE
w = " " NEW_LINE z = " " ; NEW_LINE
str = str . upper ( ) + " ▁ " ; NEW_LINE for i in range ( len ( str ) ) : NEW_LINE
ch = str [ i ] ; NEW_LINE if ( ch != ' ▁ ' ) : NEW_LINE INDENT w = w + ch ; NEW_LINE DEDENT else : NEW_LINE
z = ( z + ( w [ 0 ] ) . lower ( ) + w [ 1 : len ( w ) ] + " ▁ " ) ; NEW_LINE w = " " ; NEW_LINE return z ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; NEW_LINE print ( convert ( str ) ) ; NEW_LINE DEDENT
def countOccurrences ( str , word ) : NEW_LINE
a = str . split ( " ▁ " ) NEW_LINE
count = 0 NEW_LINE for i in range ( 0 , len ( a ) ) : NEW_LINE
if ( word == a [ i ] ) : NEW_LINE count = count + 1 NEW_LINE return count NEW_LINE
str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " NEW_LINE word = " portal " NEW_LINE print ( countOccurrences ( str , word ) ) NEW_LINE
def permute ( inp ) : NEW_LINE INDENT n = len ( inp ) NEW_LINE DEDENT
mx = 1 << n NEW_LINE
inp = inp . lower ( ) NEW_LINE
for i in range ( mx ) : NEW_LINE
combination = [ k for k in inp ] NEW_LINE for j in range ( n ) : NEW_LINE INDENT if ( ( ( i >> j ) & 1 ) == 1 ) : NEW_LINE INDENT combination [ j ] = inp [ j ] . upper ( ) NEW_LINE DEDENT DEDENT temp = " " NEW_LINE
for i in combination : NEW_LINE INDENT temp += i NEW_LINE DEDENT print temp , NEW_LINE
permute ( " ABC " ) NEW_LINE
def isPalindrome ( str ) : NEW_LINE
l = 0 NEW_LINE h = len ( str ) - 1 NEW_LINE
while ( h > l ) : NEW_LINE INDENT if ( str [ l ] != str [ h ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT l = l + 1 NEW_LINE h = h - 1 NEW_LINE DEDENT return 1 NEW_LINE
def minRemovals ( str ) : NEW_LINE
if ( str [ 0 ] == ' ' ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if ( isPalindrome ( str ) ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return 2 NEW_LINE
print ( minRemovals ( "010010" ) ) NEW_LINE print ( minRemovals ( "0100101" ) ) NEW_LINE
def power ( x , y , p ) : NEW_LINE
res = 1 ; NEW_LINE
x = x % p ; NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y and 1 ) : NEW_LINE INDENT res = ( res * x ) % p ; NEW_LINE DEDENT
y = y >> 1 ; NEW_LINE x = ( x * x ) % p ; NEW_LINE return res ; NEW_LINE
def findModuloByM ( X , N , M ) : NEW_LINE
if ( N < 6 ) : NEW_LINE
temp = chr ( 48 + X ) * N NEW_LINE
res = int ( temp ) % M ; NEW_LINE return res ; NEW_LINE
if ( N % 2 == 0 ) : NEW_LINE
half = findModuloByM ( X , N // 2 , M ) % M ; NEW_LINE
res = ( half * power ( 10 , N // 2 , M ) + half ) % M ; NEW_LINE return res ; NEW_LINE else : NEW_LINE
half = findModuloByM ( X , N // 2 , M ) % M ; NEW_LINE
res = ( half * power ( 10 , N // 2 + 1 , M ) + half * 10 + X ) % M ; NEW_LINE return res ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT X = 6 ; N = 14 ; M = 9 ; NEW_LINE DEDENT
print ( findModuloByM ( X , N , M ) ) ; NEW_LINE
def lengtang ( r1 , r2 , d ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ direct ▁ common ▁ tangent ▁ is ▁ " , ( ( d ** 2 ) - ( ( r1 - r2 ) ** 2 ) ) ** ( 1 / 2 ) ) ; NEW_LINE DEDENT
r1 = 4 ; r2 = 6 ; d = 3 ; NEW_LINE lengtang ( r1 , r2 , d ) ; NEW_LINE
def rad ( d , h ) : NEW_LINE INDENT print ( " The ▁ radius ▁ of ▁ the ▁ circle ▁ is " , ( ( d * d ) / ( 8 * h ) + h / 2 ) ) NEW_LINE DEDENT
d = 4 ; h = 1 ; NEW_LINE rad ( d , h ) ; NEW_LINE
def shortdis ( r , d ) : NEW_LINE INDENT print ( " The ▁ shortest ▁ distance ▁ " , end = " " ) ; NEW_LINE print ( " from ▁ the ▁ chord ▁ to ▁ centre ▁ " , end = " " ) ; NEW_LINE print ( ( ( r * r ) - ( ( d * d ) / 4 ) ) ** ( 1 / 2 ) ) ; NEW_LINE DEDENT
r = 4 ; NEW_LINE d = 3 ; NEW_LINE shortdis ( r , d ) ; NEW_LINE
def lengtang ( r1 , r2 , d ) : NEW_LINE INDENT print ( " The ▁ length ▁ of ▁ the ▁ direct ▁ common ▁ tangent ▁ is " , ( ( ( d ** 2 ) - ( ( r1 - r2 ) ** 2 ) ) ** ( 1 / 2 ) ) ) ; NEW_LINE DEDENT
r1 = 4 ; r2 = 6 ; d = 12 ; NEW_LINE lengtang ( r1 , r2 , d ) ; NEW_LINE
def square ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
x = 0.464 * a NEW_LINE return x NEW_LINE
a = 5 NEW_LINE print ( square ( a ) ) NEW_LINE
def polyapothem ( n , a ) : NEW_LINE
if ( a < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return a / ( 2 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 9 NEW_LINE n = 6 NEW_LINE print ( ' { 0 : . 6 } ' . format ( polyapothem ( n , a ) ) ) NEW_LINE DEDENT
def polyarea ( n , a ) : NEW_LINE
if ( a < 0 and n < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
A = ( a * a * n ) / ( 4 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) NEW_LINE return A NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 9 NEW_LINE n = 6 NEW_LINE print ( ' { 0 : . 6 } ' . format ( polyarea ( n , a ) ) ) NEW_LINE DEDENT
def calculateSide ( n , r ) : NEW_LINE INDENT theta = 360 / n NEW_LINE theta_in_radians = theta * 3.14 / 180 NEW_LINE return 2 * r * sin ( theta_in_radians / 2 ) NEW_LINE DEDENT
n = 3 NEW_LINE
r = 5 NEW_LINE print ( ' { 0 : . 5 } ' . format ( calculateSide ( n , r ) ) ) NEW_LINE
def cyl ( r , R , h ) : NEW_LINE
if ( h < 0 and r < 0 and R < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
r1 = r NEW_LINE
h1 = h NEW_LINE
V = 3.14 * pow ( r1 , 2 ) * h1 NEW_LINE return round ( V , 2 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT r , R , h = 7 , 11 , 6 NEW_LINE print ( cyl ( r , R , h ) ) NEW_LINE DEDENT
def Perimeter ( s , n ) : NEW_LINE INDENT perimeter = 1 NEW_LINE DEDENT
perimeter = n * s NEW_LINE return perimeter NEW_LINE
n = 5 NEW_LINE
s = 2.5 NEW_LINE
peri = Perimeter ( s , n ) NEW_LINE print ( " Perimeter ▁ of ▁ Regular ▁ Polygon ▁ with " , n , " sides ▁ of ▁ length " , s , " = " , peri ) NEW_LINE
def rhombusarea ( l , b ) : NEW_LINE
if ( l < 0 or b < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
return ( l * b ) / 2 NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT l = 16 NEW_LINE b = 6 NEW_LINE print ( rhombusarea ( l , b ) ) NEW_LINE DEDENT
def FindPoint ( x1 , y1 , x2 , y2 , x , y ) : NEW_LINE INDENT if ( x > x1 and x < x2 and y > y1 and y < y2 ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
x1 , y1 , x2 , y2 = 0 , 0 , 10 , 8 NEW_LINE
x , y = 1 , 5 NEW_LINE
if FindPoint ( x1 , y1 , x2 , y2 , x , y ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def shortest_distance ( x1 , y1 , z1 , a , b , c , d ) : NEW_LINE INDENT d = abs ( ( a * x1 + b * y1 + c * z1 + d ) ) NEW_LINE e = ( math . sqrt ( a * a + b * b + c * c ) ) NEW_LINE print ( " Perpendicular ▁ distance ▁ is " , d / e ) NEW_LINE DEDENT
x1 = 4 NEW_LINE y1 = - 4 NEW_LINE z1 = 3 NEW_LINE a = 2 NEW_LINE b = - 2 NEW_LINE c = 5 NEW_LINE d = 8 NEW_LINE
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) NEW_LINE
def findVolume ( l , b , h ) : NEW_LINE
return ( ( l * b * h ) / 2 ) NEW_LINE
l = 18 NEW_LINE b = 12 NEW_LINE h = 9 NEW_LINE
print ( " Volume ▁ of ▁ triangular ▁ prism : ▁ " , findVolume ( l , b , h ) ) NEW_LINE
def midpoint ( x1 , x2 , y1 , y2 ) : NEW_LINE INDENT print ( ( x1 + x2 ) // 2 , " ▁ , ▁ " , ( y1 + y2 ) // 2 ) NEW_LINE DEDENT
x1 , y1 , x2 , y2 = - 1 , 2 , 3 , - 6 NEW_LINE midpoint ( x1 , x2 , y1 , y2 ) NEW_LINE
def arcLength ( diameter , angle ) : NEW_LINE INDENT if angle >= 360 : NEW_LINE INDENT print ( " Angle ▁ cannot ▁ be ▁ formed " ) NEW_LINE return 0 NEW_LINE DEDENT else : NEW_LINE INDENT arc = ( 3.142857142857143 * diameter ) * ( angle / 360.0 ) NEW_LINE return arc NEW_LINE DEDENT DEDENT
diameter = 25.0 NEW_LINE angle = 45.0 NEW_LINE arc_len = arcLength ( diameter , angle ) NEW_LINE print ( arc_len ) NEW_LINE
import math NEW_LINE def checkCollision ( a , b , c , x , y , radius ) : NEW_LINE
dist = ( ( abs ( a * x + b * y + c ) ) / math . sqrt ( a * a + b * b ) ) NEW_LINE
if ( radius == dist ) : NEW_LINE INDENT print ( " Touch " ) NEW_LINE DEDENT elif ( radius > dist ) : NEW_LINE INDENT print ( " Intersect " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Outside " ) NEW_LINE DEDENT
radius = 5 NEW_LINE x = 0 NEW_LINE y = 0 NEW_LINE a = 3 NEW_LINE b = 4 NEW_LINE c = 25 NEW_LINE checkCollision ( a , b , c , x , y , radius ) NEW_LINE
def polygonArea ( X , Y , n ) : NEW_LINE
area = 0.0 NEW_LINE
j = n - 1 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT area = area + ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) NEW_LINE DEDENT
return abs ( area // 2.0 ) NEW_LINE
X = [ 0 , 2 , 4 ] NEW_LINE Y = [ 1 , 3 , 7 ] NEW_LINE n = len ( X ) NEW_LINE print ( polygonArea ( X , Y , n ) ) NEW_LINE
def getAverage ( x , y ) : NEW_LINE
avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; NEW_LINE return avg NEW_LINE
x = 10 NEW_LINE y = 9 NEW_LINE print ( getAverage ( x , y ) ) NEW_LINE
def smallestIndex ( a , n ) : NEW_LINE
right1 = 0 NEW_LINE right0 = 0 NEW_LINE
for i in range ( n ) : NEW_LINE
if ( a [ i ] == 1 ) : NEW_LINE INDENT right1 = i NEW_LINE DEDENT
else : NEW_LINE INDENT right0 = i NEW_LINE DEDENT
return min ( right1 , right0 ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = [ 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE n = len ( a ) NEW_LINE print ( smallestIndex ( a , n ) ) NEW_LINE DEDENT
def countSquares ( r , c , m ) : NEW_LINE
squares = 0 NEW_LINE
for i in range ( 1 , 9 ) : NEW_LINE INDENT for j in range ( 1 , 9 ) : NEW_LINE DEDENT
if ( max ( abs ( i - r ) , abs ( j - c ) ) <= m ) : NEW_LINE INDENT squares = squares + 1 NEW_LINE DEDENT
return squares NEW_LINE
r = 4 NEW_LINE c = 4 NEW_LINE m = 1 NEW_LINE print ( countSquares ( r , c , m ) ) ; NEW_LINE
def countNumbers ( L , R , K ) : NEW_LINE INDENT if ( K == 9 ) : NEW_LINE INDENT K = 0 NEW_LINE DEDENT DEDENT
totalnumbers = R - L + 1 NEW_LINE
factor9 = totalnumbers // 9 NEW_LINE
rem = totalnumbers % 9 NEW_LINE
ans = factor9 NEW_LINE
for i in range ( R , R - rem , - 1 ) : NEW_LINE INDENT rem1 = i % 9 NEW_LINE if ( rem1 == K ) : NEW_LINE INDENT ans += 1 NEW_LINE DEDENT DEDENT return ans NEW_LINE
L = 10 NEW_LINE R = 22 NEW_LINE K = 3 NEW_LINE print ( countNumbers ( L , R , K ) ) NEW_LINE
def BalanceArray ( A , Q ) : NEW_LINE INDENT ANS = [ ] NEW_LINE sum = 0 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE DEDENT
if ( A [ i ] % 2 == 0 ) : NEW_LINE INDENT sum += A [ i ] ; NEW_LINE DEDENT for i in range ( len ( Q ) ) : NEW_LINE index = Q [ i ] [ 0 ] ; NEW_LINE value = Q [ i ] [ 1 ] ; NEW_LINE
if ( A [ index ] % 2 == 0 ) : NEW_LINE INDENT sum -= A [ index ] ; NEW_LINE DEDENT A [ index ] += value ; NEW_LINE
if ( A [ index ] % 2 == 0 ) : NEW_LINE INDENT sum += A [ index ] ; NEW_LINE DEDENT
ANS . append ( sum ) ; NEW_LINE
for i in range ( len ( ANS ) ) : NEW_LINE INDENT print ( ANS [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT A = [ 1 , 2 , 3 , 4 ] ; NEW_LINE Q = [ [ 0 , 1 ] , [ 1 , - 3 ] , [ 0 , - 4 ] , [ 3 , 2 ] ] ; NEW_LINE BalanceArray ( A , Q ) ; NEW_LINE DEDENT
def Cycles ( N ) : NEW_LINE INDENT fact = 1 NEW_LINE result = N - 1 NEW_LINE DEDENT
i = result NEW_LINE while ( i > 0 ) : NEW_LINE INDENT fact = fact * i NEW_LINE i -= 1 NEW_LINE DEDENT return fact // 2 NEW_LINE
N = 5 NEW_LINE Number = Cycles ( N ) NEW_LINE print ( " Hamiltonian ▁ cycles ▁ = ▁ " , Number ) NEW_LINE
def digitWell ( n , m , k ) : NEW_LINE INDENT cnt = 0 NEW_LINE while ( n > 0 ) : NEW_LINE INDENT if ( n % 10 == m ) : NEW_LINE INDENT cnt = cnt + 1 ; NEW_LINE DEDENT n = ( int ) ( n / 10 ) ; NEW_LINE DEDENT return cnt == k ; NEW_LINE DEDENT
def findInt ( n , m , k ) : NEW_LINE INDENT i = n + 1 ; NEW_LINE while ( True ) : NEW_LINE INDENT if ( digitWell ( i , m , k ) ) : NEW_LINE INDENT return i ; NEW_LINE DEDENT i = i + 1 ; NEW_LINE DEDENT DEDENT
n = 111 ; m = 2 ; k = 2 ; NEW_LINE print ( findInt ( n , m , k ) ) ; NEW_LINE
def countOdd ( arr , n ) : NEW_LINE
odd = 0 ; NEW_LINE for i in range ( 0 , n ) : NEW_LINE
if ( arr [ i ] % 2 == 1 ) : NEW_LINE INDENT odd = odd + 1 ; NEW_LINE DEDENT return odd ; NEW_LINE
def countValidPairs ( arr , n ) : NEW_LINE INDENT odd = countOdd ( arr , n ) ; NEW_LINE return ( odd * ( odd - 1 ) ) / 2 ; NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE print ( int ( countValidPairs ( arr , n ) ) ) ; NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT if ( b == 0 ) : NEW_LINE INDENT return a NEW_LINE DEDENT else : NEW_LINE INDENT return gcd ( b , a % b ) NEW_LINE DEDENT DEDENT
def lcmOfArray ( arr , n ) : NEW_LINE INDENT if ( n < 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT lcm = arr [ 0 ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT lcm = ( lcm * arr [ i ] ) // gcd ( lcm , arr [ i ] ) ; NEW_LINE DEDENT
return lcm NEW_LINE
def minPerfectCube ( arr , n ) : NEW_LINE
lcm = lcmOfArray ( arr , n ) NEW_LINE minPerfectCube = lcm NEW_LINE cnt = 0 NEW_LINE while ( lcm > 1 and lcm % 2 == 0 ) : NEW_LINE INDENT cnt += 1 NEW_LINE lcm //= 2 NEW_LINE DEDENT
if ( cnt % 3 == 2 ) : NEW_LINE INDENT minPerfectCube *= 2 NEW_LINE DEDENT elif ( cnt % 3 == 1 ) : NEW_LINE INDENT minPerfectCube *= 4 NEW_LINE DEDENT i = 3 NEW_LINE
while ( lcm > 1 ) : NEW_LINE INDENT cnt = 0 NEW_LINE while ( lcm % i == 0 ) : NEW_LINE INDENT cnt += 1 NEW_LINE lcm //= i NEW_LINE DEDENT if ( cnt % 3 == 1 ) : NEW_LINE INDENT minPerfectCube *= i * i NEW_LINE DEDENT elif ( cnt % 3 == 2 ) : NEW_LINE INDENT minPerfectCube *= i NEW_LINE DEDENT i += 2 NEW_LINE DEDENT
return minPerfectCube NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 10 , 125 , 14 , 42 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( minPerfectCube ( arr , n ) ) NEW_LINE DEDENT
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT k = int ( sqrt ( n ) ) + 1 NEW_LINE for i in range ( 5 , k , 6 ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def isStrongPrime ( n ) : NEW_LINE
if ( isPrime ( n ) == False or n == 2 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
previous_prime = n - 1 NEW_LINE next_prime = n + 1 NEW_LINE
while ( isPrime ( next_prime ) == False ) : NEW_LINE INDENT next_prime += 1 NEW_LINE DEDENT
while ( isPrime ( previous_prime ) == False ) : NEW_LINE INDENT previous_prime -= 1 NEW_LINE DEDENT
mean = ( previous_prime + next_prime ) / 2 NEW_LINE
if ( n > mean ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 11 NEW_LINE if ( isStrongPrime ( n ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def countDigitsToBeRemoved ( N , K ) : NEW_LINE
s = str ( N ) ; NEW_LINE
res = 0 ; NEW_LINE
f_zero = 0 ; NEW_LINE for i in range ( len ( s ) - 1 , - 1 , - 1 ) : NEW_LINE INDENT if ( K == 0 ) : NEW_LINE INDENT return res ; NEW_LINE DEDENT if ( s [ i ] == '0' ) : NEW_LINE DEDENT
f_zero = 1 ; NEW_LINE K -= 1 ; NEW_LINE else : NEW_LINE res += 1 ; NEW_LINE
if ( K == 0 ) : NEW_LINE INDENT return res ; NEW_LINE DEDENT elif ( f_zero > 0 ) : NEW_LINE INDENT return len ( s ) - 1 ; NEW_LINE DEDENT return - 1 ; NEW_LINE
N = 10904025 ; NEW_LINE K = 2 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE N = 1000 ; NEW_LINE K = 5 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE N = 23985 ; NEW_LINE K = 2 ; NEW_LINE print ( countDigitsToBeRemoved ( N , K ) ) ; NEW_LINE
def getSum ( a , n ) : NEW_LINE
sum = 0 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE
sum += ( i / math . pow ( a , i ) ) ; NEW_LINE return sum ; NEW_LINE
a = 3 ; n = 3 ; NEW_LINE
print ( getSum ( a , n ) ) ; NEW_LINE
def largestPrimeFactor ( n ) : NEW_LINE
max = - 1 NEW_LINE
while n % 2 == 0 : NEW_LINE INDENT max = 2 ; NEW_LINE DEDENT
for i in range ( 3 , int ( sqrt ( n ) ) + 1 , 2 ) : NEW_LINE INDENT while n % i == 0 : NEW_LINE INDENT max = i ; NEW_LINE n = n / i ; NEW_LINE DEDENT DEDENT
if n > 2 : NEW_LINE INDENT max = n NEW_LINE DEDENT return max NEW_LINE
def checkUnusual ( n ) : NEW_LINE
factor = largestPrimeFactor ( n ) NEW_LINE
if factor > sqrt ( n ) : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 14 NEW_LINE if checkUnusual ( n ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT DEDENT
def isHalfReducible ( arr , n , m ) : NEW_LINE INDENT frequencyHash = [ 0 ] * ( m + 1 ) ; NEW_LINE i = 0 ; NEW_LINE while ( i < n ) : NEW_LINE INDENT frequencyHash [ ( arr [ i ] % ( m + 1 ) ) ] += 1 ; NEW_LINE i += 1 ; NEW_LINE DEDENT i = 0 ; NEW_LINE while ( i <= m ) : NEW_LINE INDENT if ( frequencyHash [ i ] >= ( n / 2 ) ) : NEW_LINE INDENT break ; NEW_LINE DEDENT i += 1 ; NEW_LINE DEDENT if ( i <= m ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT DEDENT
arr = [ 8 , 16 , 32 , 3 , 12 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE m = 7 ; NEW_LINE isHalfReducible ( arr , n , m ) ; NEW_LINE
def isPrime ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return True NEW_LINE
def isPowerOfTwo ( n ) : NEW_LINE INDENT return ( n and ( not ( n & ( n - 1 ) ) ) ) NEW_LINE DEDENT
n = 43 NEW_LINE
if ( isPrime ( n ) and isPowerOfTwo ( n * 3 - 1 ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def area ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
area = pow ( ( a * sqrt ( 3 ) ) / ( sqrt ( 2 ) ) , 2 ) NEW_LINE return area NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT a = 5 NEW_LINE print ( " { 0 : . 3 } " . format ( area ( a ) ) ) NEW_LINE DEDENT
def nthTerm ( n ) : NEW_LINE INDENT return 3 * pow ( n , 2 ) - 4 * n + 2 NEW_LINE DEDENT
N = 4 NEW_LINE print ( nthTerm ( N ) ) NEW_LINE
def calculateSum ( n ) : NEW_LINE INDENT return ( n * ( n + 1 ) // 2 + pow ( ( n * ( n + 1 ) // 2 ) , 2 ) ) NEW_LINE DEDENT
n = 3 NEW_LINE
print ( " Sum ▁ = ▁ " , calculateSum ( n ) ) NEW_LINE
def arePermutations ( a , b , n , m ) : NEW_LINE INDENT sum1 , sum2 , mul1 , mul2 = 0 , 0 , 1 , 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT sum1 += a [ i ] NEW_LINE mul1 *= a [ i ] NEW_LINE DEDENT
for i in range ( m ) : NEW_LINE INDENT sum2 += b [ i ] NEW_LINE mul2 *= b [ i ] NEW_LINE DEDENT
return ( ( sum1 == sum2 ) and ( mul1 == mul2 ) ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = [ 1 , 3 , 2 ] NEW_LINE b = [ 3 , 1 , 2 ] NEW_LINE n = len ( a ) NEW_LINE m = len ( b ) NEW_LINE if arePermutations ( a , b , n , m ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT DEDENT
def Race ( B , C ) : NEW_LINE INDENT result = 0 ; NEW_LINE DEDENT
result = ( ( C * 100 ) // B ) NEW_LINE return 100 - result NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT B = 10 NEW_LINE C = 28 NEW_LINE DEDENT
B = 100 - B ; NEW_LINE C = 100 - C ; NEW_LINE print ( str ( Race ( B , C ) ) + " ▁ meters " ) NEW_LINE
def Time ( arr , n , Emptypipe ) : NEW_LINE INDENT fill = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT fill += ( 1 / arr [ i ] ) NEW_LINE DEDENT fill = fill - ( 1 / float ( Emptypipe ) ) NEW_LINE return int ( 1 / fill ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 12 , 14 ] NEW_LINE Emptypipe = 30 NEW_LINE n = len ( arr ) NEW_LINE print ( ( Time ( arr , n , Emptypipe ) ) , " Hours " ) NEW_LINE DEDENT
def check ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE DEDENT
while n != 0 : NEW_LINE INDENT sum += n % 10 NEW_LINE n = n // 10 NEW_LINE DEDENT
if sum % 7 == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT else : NEW_LINE INDENT return 0 NEW_LINE DEDENT
n = 25 NEW_LINE print ( ( " YES " ) if check ( n ) == 1 else print ( " NO " ) ) NEW_LINE
N = 1000005 NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT if n <= 3 : NEW_LINE INDENT return True NEW_LINE DEDENT
if n % 2 == 0 or n % 3 == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT i = 5 NEW_LINE while i * i <= n : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return True NEW_LINE
def SumOfPrimeDivisors ( n ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT if isPrime ( i ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT DEDENT DEDENT return sum NEW_LINE DEDENT
n = 60 NEW_LINE print ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + str ( SumOfPrimeDivisors ( n ) ) ) NEW_LINE
def Sum ( N ) : NEW_LINE INDENT SumOfPrimeDivisors = [ 0 ] * ( N + 1 ) NEW_LINE for i in range ( 2 , N + 1 ) : NEW_LINE DEDENT
if ( SumOfPrimeDivisors [ i ] == 0 ) : NEW_LINE
for j in range ( i , N + 1 , i ) : NEW_LINE INDENT SumOfPrimeDivisors [ j ] += i NEW_LINE DEDENT return SumOfPrimeDivisors [ N ] NEW_LINE
N = 60 NEW_LINE print ( " Sum ▁ of ▁ prime " , " divisors ▁ of ▁ 60 ▁ is " , Sum ( N ) ) ; NEW_LINE
def power ( x , y , p ) : NEW_LINE
x = x % p NEW_LINE while ( y > 0 ) : NEW_LINE
if ( y & 1 ) : NEW_LINE INDENT res = ( res * x ) % p NEW_LINE DEDENT
x = ( x * x ) % p NEW_LINE return res NEW_LINE
a = 3 NEW_LINE
b = "100000000000000000000000000" NEW_LINE remainderB = 0 NEW_LINE MOD = 1000000007 NEW_LINE
for i in range ( len ( b ) ) : NEW_LINE INDENT remainderB = ( ( remainderB * 10 + ord ( b [ i ] ) - 48 ) % ( MOD - 1 ) ) NEW_LINE DEDENT print ( power ( a , remainderB , MOD ) ) NEW_LINE
def find_Square_369 ( num ) : NEW_LINE
if ( num [ 0 ] == '3' ) : NEW_LINE INDENT a = '1' NEW_LINE b = '0' NEW_LINE c = '8' NEW_LINE d = '9' NEW_LINE DEDENT
elif ( num [ 0 ] == '6' ) : NEW_LINE INDENT a = '4' NEW_LINE b = '3' NEW_LINE c = '5' NEW_LINE d = '6' NEW_LINE DEDENT
else : NEW_LINE INDENT a = '9' NEW_LINE b = '8' NEW_LINE c = '0' NEW_LINE d = '1' NEW_LINE DEDENT
result = " " NEW_LINE
size = len ( num ) NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT result += a NEW_LINE DEDENT
result += b NEW_LINE
for i in range ( 1 , size ) : NEW_LINE INDENT result += c NEW_LINE DEDENT
result += d NEW_LINE
return result NEW_LINE
num_3 = "3333" NEW_LINE num_6 = "6666" NEW_LINE num_9 = "9999" NEW_LINE result = " " NEW_LINE
result = find_Square_369 ( num_3 ) NEW_LINE print ( " Square ▁ of ▁ " + num_3 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
result = find_Square_369 ( num_6 ) NEW_LINE print ( " Square ▁ of ▁ " + num_6 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
result = find_Square_369 ( num_9 ) NEW_LINE print ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result ) ; NEW_LINE
ans = 1 NEW_LINE mod = 1000000007 * 120 NEW_LINE for i in range ( 0 , 5 ) : NEW_LINE INDENT ans = ( ans * ( 55555 - i ) ) % mod NEW_LINE DEDENT ans = int ( ans / 120 ) NEW_LINE print ( " Answer ▁ using ▁ shortcut : ▁ " , ans ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 or n == 1 ) : NEW_LINE INDENT return 1 ; NEW_LINE DEDENT ans = 1 ; NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT ans = ans * i ; NEW_LINE DEDENT return ans ; NEW_LINE DEDENT
def nCr ( n , r ) : NEW_LINE INDENT Nr = n ; Dr = 1 ; ans = 1 ; NEW_LINE for i in range ( 1 , r + 1 ) : NEW_LINE INDENT ans = int ( ( ans * Nr ) / ( Dr ) ) ; NEW_LINE Nr = Nr - 1 ; NEW_LINE Dr = Dr + 1 ; NEW_LINE DEDENT return ans ; NEW_LINE DEDENT
def solve ( n ) : NEW_LINE INDENT N = 2 * n - 2 ; NEW_LINE R = n - 1 ; NEW_LINE return ( nCr ( N , R ) * fact ( n - 1 ) ) ; NEW_LINE DEDENT
n = 6 ; NEW_LINE print ( solve ( n ) ) ; NEW_LINE
def pythagoreanTriplet ( n ) : NEW_LINE
for i in range ( 1 , int ( n / 3 ) + 1 ) : NEW_LINE
for j in range ( i + 1 , int ( n / 2 ) + 1 ) : NEW_LINE INDENT k = n - i - j NEW_LINE if ( i * i + j * j == k * k ) : NEW_LINE INDENT print ( i , " , ▁ " , j , " , ▁ " , k , sep = " " ) NEW_LINE return NEW_LINE DEDENT DEDENT print ( " No ▁ Triplet " ) NEW_LINE
n = 12 NEW_LINE pythagoreanTriplet ( n ) NEW_LINE
def factorial ( n ) : NEW_LINE INDENT f = 1 NEW_LINE for i in range ( 2 , n + 1 ) : NEW_LINE INDENT f *= i NEW_LINE DEDENT return f NEW_LINE DEDENT
def series ( A , X , n ) : NEW_LINE
nFact = factorial ( n ) NEW_LINE
for i in range ( 0 , n + 1 ) : NEW_LINE
niFact = factorial ( n - i ) NEW_LINE iFact = factorial ( i ) NEW_LINE
aPow = pow ( A , n - i ) NEW_LINE xPow = pow ( X , i ) NEW_LINE
print ( int ( ( nFact * aPow * xPow ) / ( niFact * iFact ) ) , end = " ▁ " ) NEW_LINE
A = 3 ; X = 4 ; n = 5 NEW_LINE series ( A , X , n ) NEW_LINE
def seiresSum ( n , a ) : NEW_LINE INDENT res = 0 NEW_LINE for i in range ( 0 , 2 * n ) : NEW_LINE INDENT if ( i % 2 == 0 ) : NEW_LINE INDENT res += a [ i ] * a [ i ] NEW_LINE DEDENT else : NEW_LINE INDENT res -= a [ i ] * a [ i ] NEW_LINE DEDENT DEDENT return res NEW_LINE DEDENT
n = 2 NEW_LINE a = [ 1 , 2 , 3 , 4 ] NEW_LINE print ( seiresSum ( n , a ) ) NEW_LINE
def power ( n , r ) : NEW_LINE
count = 0 ; i = r NEW_LINE while ( ( n / i ) >= 1 ) : NEW_LINE INDENT count += n / i NEW_LINE i = i * r NEW_LINE DEDENT return int ( count ) NEW_LINE
n = 6 ; r = 3 NEW_LINE print ( power ( n , r ) ) NEW_LINE
def avg_of_odd_num ( n ) : NEW_LINE
sm = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sm = sm + ( 2 * i + 1 ) NEW_LINE DEDENT
return sm // n NEW_LINE
n = 20 NEW_LINE print ( avg_of_odd_num ( n ) ) NEW_LINE
def avg_of_odd_num ( n ) : NEW_LINE INDENT return n NEW_LINE DEDENT
n = 8 NEW_LINE print ( avg_of_odd_num ( n ) ) NEW_LINE
def fib ( f , N ) : NEW_LINE
f [ 1 ] = 1 NEW_LINE f [ 2 ] = 1 NEW_LINE for i in range ( 3 , N + 1 ) : NEW_LINE
f [ i ] = f [ i - 1 ] + f [ i - 2 ] NEW_LINE def fiboTriangle ( n ) : NEW_LINE
N = n * ( n + 1 ) // 2 NEW_LINE f = [ 0 ] * ( N + 1 ) NEW_LINE fib ( f , N ) NEW_LINE
fiboNum = 1 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( f [ fiboNum ] , " ▁ " , end = " " ) NEW_LINE fiboNum = fiboNum + 1 NEW_LINE DEDENT print ( ) NEW_LINE
n = 5 NEW_LINE fiboTriangle ( n ) NEW_LINE
def averageOdd ( n ) : NEW_LINE INDENT if ( n % 2 == 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT sm = 0 NEW_LINE count = 0 NEW_LINE while ( n >= 1 ) : NEW_LINE DEDENT
count = count + 1 NEW_LINE
sm = sm + n NEW_LINE n = n - 2 NEW_LINE return sm // count NEW_LINE
n = 15 NEW_LINE print ( averageOdd ( n ) ) NEW_LINE
def averageOdd ( n ) : NEW_LINE INDENT if ( n % 2 == 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 1 ) // 2 NEW_LINE DEDENT
n = 15 NEW_LINE print ( averageOdd ( n ) ) NEW_LINE
def TrinomialValue ( n , k ) : NEW_LINE
if n == 0 and k == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if k < - n or k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return ( TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ) NEW_LINE
def printTrinomial ( n ) : NEW_LINE
for i in range ( n ) : NEW_LINE
for j in range ( - i , 1 ) : NEW_LINE INDENT print ( TrinomialValue ( i , j ) , end = " ▁ " ) NEW_LINE DEDENT
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( TrinomialValue ( i , j ) , end = " ▁ " ) NEW_LINE DEDENT print ( " " , end = ' ' ) NEW_LINE
n = 4 NEW_LINE printTrinomial ( n ) NEW_LINE
def TrinomialValue ( dp , n , k ) : NEW_LINE
if k < 0 : NEW_LINE INDENT k = - k NEW_LINE DEDENT
if dp [ n ] [ k ] != 0 : NEW_LINE INDENT return dp [ n ] [ k ] NEW_LINE DEDENT
if n == 0 and k == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if k < - n or k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return ( TrinomialValue ( dp , n - 1 , k - 1 ) + TrinomialValue ( dp , n - 1 , k ) + TrinomialValue ( dp , n - 1 , k + 1 ) ) NEW_LINE
def printTrinomial ( n ) : NEW_LINE INDENT dp = [ [ 0 ] * 10 ] * 10 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE
for j in range ( - i , 1 ) : NEW_LINE INDENT print ( TrinomialValue ( dp , i , j ) , end = " ▁ " ) NEW_LINE DEDENT
for j in range ( 1 , i + 1 ) : NEW_LINE INDENT print ( TrinomialValue ( dp , i , j ) , end = " ▁ " ) NEW_LINE DEDENT print ( " " , end = ' ' ) NEW_LINE
n = 4 NEW_LINE printTrinomial ( n ) NEW_LINE
def sumOfLargePrimeFactor ( n ) : NEW_LINE
prime = [ 0 ] * ( n + 1 ) NEW_LINE sum = 0 NEW_LINE max = int ( n / 2 ) NEW_LINE for p in range ( 2 , max + 1 ) : NEW_LINE
if prime [ p ] == 0 : NEW_LINE
for i in range ( p * 2 , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = p NEW_LINE DEDENT
for p in range ( 2 , n + 1 ) : NEW_LINE
if prime [ p ] : NEW_LINE INDENT sum += prime [ p ] NEW_LINE DEDENT
else : NEW_LINE INDENT sum += p NEW_LINE DEDENT
return sum NEW_LINE
n = 12 NEW_LINE print ( " Sum ▁ = " , sumOfLargePrimeFactor ( n ) ) NEW_LINE
def calculate_sum ( a , N ) : NEW_LINE
m = N / a NEW_LINE
sum = m * ( m + 1 ) / 2 NEW_LINE
ans = a * sum NEW_LINE print ( " Sum ▁ of ▁ multiples ▁ of ▁ " , a , " ▁ up ▁ to ▁ " , N , " ▁ = ▁ " , ans ) NEW_LINE
calculate_sum ( 7 , 49 ) NEW_LINE
def isPowerOf2 ( sttr ) : NEW_LINE INDENT len_str = len ( sttr ) ; NEW_LINE sttr = list ( sttr ) ; NEW_LINE DEDENT
num = 0 ; NEW_LINE
if ( len_str == 1 and sttr [ len_str - 1 ] == '1' ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
while ( len_str != 1 or sttr [ len_str - 1 ] != '1' ) : NEW_LINE
if ( ( ord ( sttr [ len_str - 1 ] ) - ord ( '0' ) ) % 2 == 1 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
j = 0 ; NEW_LINE for i in range ( len_str ) : NEW_LINE INDENT num = num * 10 + ( ord ( sttr [ i ] ) - ord ( '0' ) ) ; NEW_LINE DEDENT
if ( num < 2 ) : NEW_LINE
if ( i != 0 ) : NEW_LINE INDENT sttr [ j ] = '0' ; NEW_LINE j += 1 ; NEW_LINE DEDENT
continue ; NEW_LINE sttr [ j ] = chr ( ( num // 2 ) + ord ( '0' ) ) ; NEW_LINE j += 1 ; NEW_LINE num = ( num ) - ( num // 2 ) * 2 ; NEW_LINE
len_str = j ; NEW_LINE
return 1 ; NEW_LINE
str1 = "124684622466842024680246842024662202000002" ; NEW_LINE str2 = "1" ; NEW_LINE str3 = "128" ; NEW_LINE print ( " " , isPowerOf2 ( str1 ) ,   " " , isPowerOf2 ( str2 ) ,   " " , isPowerOf2 ( str3 ) ) ; NEW_LINE
def ispowerof2 ( num ) : NEW_LINE INDENT if ( ( num & ( num - 1 ) ) == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return 0 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT num = 549755813888 NEW_LINE print ( ispowerof2 ( num ) ) NEW_LINE DEDENT
def counDivisors ( X ) : NEW_LINE
count = 0 NEW_LINE
for i in range ( 1 , X + 1 ) : NEW_LINE INDENT if ( X % i == 0 ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT
return count NEW_LINE
def countDivisorsMult ( arr , n ) : NEW_LINE
mul = 1 NEW_LINE for i in range ( n ) : NEW_LINE INDENT mul *= arr [ i ] NEW_LINE DEDENT
return counDivisors ( mul ) NEW_LINE
arr = [ 2 , 4 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( countDivisorsMult ( arr , n ) ) NEW_LINE
def freqPairs ( arr , n ) : NEW_LINE
max = arr [ 0 ] NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT if arr [ i ] > max : NEW_LINE INDENT max = arr [ i ] NEW_LINE DEDENT DEDENT
freq = [ 0 for i in range ( max + 1 ) ] NEW_LINE
count = 0 NEW_LINE
for i in range ( n ) : NEW_LINE INDENT freq [ arr [ i ] ] += 1 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT for j in range ( 2 * arr [ i ] , max + 1 , arr [ i ] ) : NEW_LINE DEDENT
if ( freq [ j ] >= 1 ) : NEW_LINE INDENT count += freq [ j ] NEW_LINE DEDENT
if ( freq [ arr [ i ] ] > 1 ) : NEW_LINE INDENT count += freq [ arr [ i ] ] - 1 NEW_LINE freq [ arr [ i ] ] -= 1 NEW_LINE DEDENT return count NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ 3 , 2 , 4 , 2 , 6 ] NEW_LINE n = len ( arr ) NEW_LINE print ( freqPairs ( arr , n ) ) NEW_LINE DEDENT
def Nth_Term ( n ) : NEW_LINE INDENT return ( 2 * pow ( n , 3 ) - 3 * pow ( n , 2 ) + n + 6 ) // 6 NEW_LINE DEDENT
N = 8 NEW_LINE print ( Nth_Term ( N ) ) NEW_LINE
def printNthElement ( n ) : NEW_LINE
arr = [ 0 ] * ( n + 1 ) ; NEW_LINE arr [ 1 ] = 3 NEW_LINE arr [ 2 ] = 5 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE
if ( i % 2 != 0 ) : NEW_LINE INDENT arr [ i ] = arr [ i // 2 ] * 10 + 3 NEW_LINE DEDENT else : NEW_LINE INDENT arr [ i ] = arr [ ( i // 2 ) - 1 ] * 10 + 5 NEW_LINE DEDENT return arr [ n ] NEW_LINE
n = 6 NEW_LINE print ( printNthElement ( n ) ) NEW_LINE
def nthTerm ( N ) : NEW_LINE
return ( N * ( ( N // 2 ) + ( ( N % 2 ) * 2 ) + N ) ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 5 NEW_LINE DEDENT
print ( " Nth ▁ term ▁ for ▁ N ▁ = ▁ " , N , " ▁ : ▁ " , nthTerm ( N ) ) NEW_LINE
def series ( A , X , n ) : NEW_LINE
term = pow ( A , n ) NEW_LINE print ( term , end = " ▁ " ) NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE
term = int ( term * X * ( n - i + 1 ) / ( i * A ) ) NEW_LINE print ( term , end = " ▁ " ) NEW_LINE
A = 3 ; X = 4 ; n = 5 NEW_LINE series ( A , X , n ) NEW_LINE
def Div_by_8 ( n ) : NEW_LINE INDENT return ( ( ( n >> 3 ) << 3 ) == n ) NEW_LINE DEDENT
n = 16 NEW_LINE if ( Div_by_8 ( n ) ) : NEW_LINE INDENT print ( " YES " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NO " ) NEW_LINE DEDENT
def averageEven ( n ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT sm = 0 NEW_LINE count = 0 NEW_LINE while ( n >= 2 ) : NEW_LINE DEDENT
count = count + 1 NEW_LINE
sm = sm + n NEW_LINE n = n - 2 NEW_LINE return sm // count NEW_LINE
n = 16 NEW_LINE print ( averageEven ( n ) ) NEW_LINE
def averageEven ( n ) : NEW_LINE INDENT if ( n % 2 != 0 ) : NEW_LINE INDENT print ( " Invalid ▁ Input " ) NEW_LINE return - 1 NEW_LINE DEDENT return ( n + 2 ) // 2 NEW_LINE DEDENT
n = 16 NEW_LINE print ( averageEven ( n ) ) NEW_LINE
def gcd ( a , b ) : NEW_LINE
if a == 0 or b == 0 : NEW_LINE INDENT return 0 NEW_LINE DEDENT
if a == b : NEW_LINE INDENT return a NEW_LINE DEDENT
if a > b : NEW_LINE INDENT return gcd ( a - b , b ) NEW_LINE DEDENT return gcd ( a , b - a ) NEW_LINE
def cpFact ( x , y ) : NEW_LINE INDENT while gcd ( x , y ) != 1 : NEW_LINE INDENT x = x / gcd ( x , y ) NEW_LINE DEDENT return int ( x ) NEW_LINE DEDENT
x = 15 NEW_LINE y = 3 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE x = 14 NEW_LINE y = 28 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE x = 7 NEW_LINE y = 3 NEW_LINE print ( cpFact ( x , y ) ) NEW_LINE
def counLastDigitK ( low , high , k ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( low , high + 1 ) : NEW_LINE INDENT if ( i % 10 == k ) : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT return count NEW_LINE DEDENT
low = 3 NEW_LINE high = 35 NEW_LINE k = 3 NEW_LINE print ( counLastDigitK ( low , high , k ) ) NEW_LINE
import math NEW_LINE def printTaxicab2 ( N ) : NEW_LINE
i , count = 1 , 0 NEW_LINE while ( count < N ) : NEW_LINE INDENT int_count = 0 NEW_LINE DEDENT
for j in range ( 1 , math . ceil ( pow ( i , 1.0 / 3 ) ) + 1 ) : NEW_LINE INDENT for k in range ( j + 1 , math . ceil ( pow ( i , 1.0 / 3 ) ) + 1 ) : NEW_LINE INDENT if ( j * j * j + k * k * k == i ) : NEW_LINE INDENT int_count += 1 NEW_LINE DEDENT DEDENT DEDENT
if ( int_count == 2 ) : NEW_LINE INDENT count += 1 NEW_LINE print ( count , " ▁ " , i ) NEW_LINE DEDENT i += 1 NEW_LINE
N = 5 NEW_LINE printTaxicab2 ( N ) NEW_LINE
def isComposite ( n ) : NEW_LINE
if ( n <= 1 ) : NEW_LINE INDENT return False NEW_LINE DEDENT if ( n <= 3 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( n % 2 == 0 or n % 3 == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = 5 NEW_LINE while ( i * i <= n ) : NEW_LINE INDENT if ( n % i == 0 or n % ( i + 2 ) == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT i = i + 6 NEW_LINE DEDENT return False NEW_LINE
print ( " true " ) if ( isComposite ( 11 ) ) else print ( " false " ) NEW_LINE print ( " true " ) if ( isComposite ( 15 ) ) else print ( " false " ) NEW_LINE
def isPrime ( n ) : NEW_LINE
if n <= 1 : NEW_LINE INDENT return False NEW_LINE DEDENT
for i in range ( 2 , n ) : NEW_LINE INDENT if n % i == 0 : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
def findPrime ( n ) : NEW_LINE INDENT num = n + 1 NEW_LINE DEDENT
while ( num ) : NEW_LINE
if isPrime ( num ) : NEW_LINE INDENT return num NEW_LINE DEDENT
num += 1 NEW_LINE return 0 NEW_LINE
def minNumber ( arr ) : NEW_LINE INDENT s = 0 NEW_LINE DEDENT
for i in range ( 0 , len ( arr ) ) : NEW_LINE INDENT s += arr [ i ] NEW_LINE DEDENT
if isPrime ( s ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
num = findPrime ( s ) NEW_LINE
return num - s NEW_LINE
arr = [ 2 , 4 , 6 , 8 , 12 ] NEW_LINE print ( minNumber ( arr ) ) NEW_LINE
def fact ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * fact ( n - 1 ) NEW_LINE DEDENT
def div ( x ) : NEW_LINE INDENT ans = 0 ; NEW_LINE for i in range ( 1 , x + 1 ) : NEW_LINE INDENT if ( x % i == 0 ) : NEW_LINE INDENT ans += i NEW_LINE DEDENT DEDENT return ans NEW_LINE DEDENT
def sumFactDiv ( n ) : NEW_LINE INDENT return div ( fact ( n ) ) NEW_LINE DEDENT
n = 4 NEW_LINE print ( sumFactDiv ( n ) ) NEW_LINE
allPrimes = [ ] ; NEW_LINE
def sieve ( n ) : NEW_LINE
prime = [ True ] * ( n + 1 ) ; NEW_LINE
p = 2 ; NEW_LINE while ( p * p <= n ) : NEW_LINE
if ( prime [ p ] == True ) : NEW_LINE
for i in range ( p * 2 , n + 1 , p ) : NEW_LINE INDENT prime [ i ] = False ; NEW_LINE DEDENT p += 1 ; NEW_LINE
for p in range ( 2 , n + 1 ) : NEW_LINE INDENT if ( prime [ p ] ) : NEW_LINE INDENT allPrimes . append ( p ) ; NEW_LINE DEDENT DEDENT
def factorialDivisors ( n ) : NEW_LINE
result = 1 ; NEW_LINE
for i in range ( len ( allPrimes ) ) : NEW_LINE
p = allPrimes [ i ] ; NEW_LINE
exp = 0 ; NEW_LINE while ( p <= n ) : NEW_LINE INDENT exp = exp + int ( n / p ) ; NEW_LINE p = p * allPrimes [ i ] ; NEW_LINE DEDENT
result = int ( result * ( pow ( allPrimes [ i ] , exp + 1 ) - 1 ) / ( allPrimes [ i ] - 1 ) ) ; NEW_LINE
return result ; NEW_LINE
print ( factorialDivisors ( 4 ) ) ; NEW_LINE
def checkPandigital ( b , n ) : NEW_LINE
if ( len ( n ) < b ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT hash = [ 0 ] * b ; NEW_LINE
for i in range ( len ( n ) ) : NEW_LINE
if ( n [ i ] >= '0' and n [ i ] <= '9' ) : NEW_LINE INDENT hash [ ord ( n [ i ] ) - ord ( '0' ) ] = 1 ; NEW_LINE DEDENT
elif ( ord ( n [ i ] ) - ord ( ' A ' ) <= b - 11 ) : NEW_LINE INDENT hash [ ord ( n [ i ] ) - ord ( ' A ' ) + 10 ] = 1 ; NEW_LINE DEDENT
for i in range ( b ) : NEW_LINE INDENT if ( hash [ i ] == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT DEDENT return 1 ; NEW_LINE
b = 13 ; NEW_LINE n = "1298450376ABC " ; NEW_LINE if ( checkPandigital ( b , n ) ) : NEW_LINE INDENT print ( " Yes " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) ; NEW_LINE DEDENT
def conver ( m , n ) : NEW_LINE INDENT if ( m == n ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT DEDENT
if ( m > n ) : NEW_LINE INDENT return m - n NEW_LINE DEDENT
if ( m <= 0 and n > 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
if ( n % 2 == 1 ) : NEW_LINE
return 1 + conver ( m , n + 1 ) NEW_LINE
else : NEW_LINE
return 1 + conver ( m , n / 2 ) NEW_LINE
m = 3 NEW_LINE n = 11 NEW_LINE print ( " Minimum ▁ number ▁ of ▁ operations ▁ : " , conver ( m , n ) ) NEW_LINE
MAX = 10000 ; NEW_LINE prodDig = [ 0 ] * MAX ; NEW_LINE
def getDigitProduct ( x ) : NEW_LINE
if ( x < 10 ) : NEW_LINE INDENT return x ; NEW_LINE DEDENT
if ( prodDig [ x ] != 0 ) : NEW_LINE INDENT return prodDig [ x ] ; NEW_LINE DEDENT
prod = ( int ( x % 10 ) * getDigitProduct ( int ( x / 10 ) ) ) ; NEW_LINE prodDig [ x ] = prod ; NEW_LINE return prod ; NEW_LINE
def findSeed ( n ) : NEW_LINE
res = [ ] ; NEW_LINE for i in range ( 1 , int ( n / 2 + 2 ) ) : NEW_LINE INDENT if ( i * getDigitProduct ( i ) == n ) : NEW_LINE INDENT res . append ( i ) ; NEW_LINE DEDENT DEDENT
if ( len ( res ) == 0 ) : NEW_LINE INDENT print ( " NO ▁ seed ▁ exists " ) ; NEW_LINE return ; NEW_LINE DEDENT
for i in range ( len ( res ) ) : NEW_LINE INDENT print ( res [ i ] , end = " ▁ " ) ; NEW_LINE DEDENT
n = 138 ; NEW_LINE findSeed ( n ) ; NEW_LINE
def maxPrimefactorNum ( N ) : NEW_LINE INDENT arr = [ 0 for i in range ( N + 5 ) ] NEW_LINE DEDENT
for i in range ( 2 , int ( sqrt ( N ) ) + 1 , 1 ) : NEW_LINE INDENT if ( arr [ i ] == 0 ) : NEW_LINE INDENT for j in range ( 2 * i , N + 1 , i ) : NEW_LINE INDENT arr [ j ] += 1 NEW_LINE DEDENT DEDENT arr [ i ] = 1 NEW_LINE DEDENT maxval = 0 NEW_LINE maxint = 1 NEW_LINE
for i in range ( 1 , N + 1 , 1 ) : NEW_LINE INDENT if ( arr [ i ] > maxval ) : NEW_LINE INDENT maxval = arr [ i ] NEW_LINE maxint = i NEW_LINE DEDENT DEDENT return maxint NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT N = 40 NEW_LINE print ( maxPrimefactorNum ( N ) ) NEW_LINE DEDENT
def SubArraySum ( arr , n ) : NEW_LINE INDENT result = 0 NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) NEW_LINE DEDENT
return result NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Sum ▁ of ▁ SubArray ▁ : ▁ " , SubArraySum ( arr , n ) ) NEW_LINE
def highestPowerof2 ( n ) : NEW_LINE INDENT res = 0 ; NEW_LINE for i in range ( n , 0 , - 1 ) : NEW_LINE DEDENT
if ( ( i & ( i - 1 ) ) == 0 ) : NEW_LINE INDENT res = i ; NEW_LINE break ; NEW_LINE DEDENT return res ; NEW_LINE
n = 10 ; NEW_LINE print ( highestPowerof2 ( n ) ) ; NEW_LINE
def findPairs ( n ) : NEW_LINE
cubeRoot = int ( math . pow ( n , 1.0 / 3.0 ) ) ; NEW_LINE
' NEW_LINE INDENT cube = [ 0 ] * ( cubeRoot + 1 ) ; NEW_LINE DEDENT
for i in range ( 1 , cubeRoot + 1 ) : NEW_LINE INDENT cube [ i ] = i * i * i ; NEW_LINE DEDENT
l = 1 ; NEW_LINE r = cubeRoot ; NEW_LINE while ( l < r ) : NEW_LINE INDENT if ( cube [ l ] + cube [ r ] < n ) : NEW_LINE INDENT l += 1 ; NEW_LINE DEDENT elif ( cube [ l ] + cube [ r ] > n ) : NEW_LINE INDENT r -= 1 ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ( " , l , " , ▁ " , math . floor ( r ) , " ) " , end = " " ) ; NEW_LINE print ( ) ; NEW_LINE l += 1 ; NEW_LINE r -= 1 ; NEW_LINE DEDENT DEDENT
n = 20683 ; NEW_LINE findPairs ( n ) ; NEW_LINE
def gcd ( a , b ) : NEW_LINE INDENT while ( b != 0 ) : NEW_LINE INDENT t = b NEW_LINE b = a % b NEW_LINE a = t NEW_LINE DEDENT return a NEW_LINE DEDENT
def findMinDiff ( a , b , x , y ) : NEW_LINE
g = gcd ( a , b ) NEW_LINE
diff = abs ( x - y ) % g NEW_LINE return min ( diff , g - diff ) NEW_LINE
a , b , x , y = 20 , 52 , 5 , 7 NEW_LINE print ( findMinDiff ( a , b , x , y ) ) NEW_LINE
def printDivisors ( n ) : NEW_LINE INDENT list = [ ] NEW_LINE DEDENT
for i in range ( 1 , int ( math . sqrt ( n ) + 1 ) ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( n / i == i ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT else : NEW_LINE
print ( i , end = " ▁ " ) NEW_LINE list . append ( int ( n / i ) ) NEW_LINE
for i in list [ : : - 1 ] : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE DEDENT
print ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) NEW_LINE printDivisors ( 100 ) NEW_LINE
def printDivisors ( n ) : NEW_LINE INDENT i = 1 NEW_LINE while i <= n : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE INDENT print i , NEW_LINE DEDENT i = i + 1 NEW_LINE DEDENT DEDENT
print " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " NEW_LINE printDivisors ( 100 ) NEW_LINE
def printDivisors ( n ) : NEW_LINE
i = 1 NEW_LINE while i <= math . sqrt ( n ) : NEW_LINE INDENT if ( n % i == 0 ) : NEW_LINE DEDENT
if ( n / i == i ) : NEW_LINE INDENT print i , NEW_LINE DEDENT else : NEW_LINE
print i , n / i , NEW_LINE i = i + 1 NEW_LINE
print " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " NEW_LINE printDivisors ( 100 ) NEW_LINE
def SieveOfAtkin ( limit ) : NEW_LINE
if ( limit > 2 ) : NEW_LINE INDENT print ( 2 , end = " ▁ " ) NEW_LINE DEDENT if ( limit > 3 ) : NEW_LINE INDENT print ( 3 , end = " ▁ " ) NEW_LINE DEDENT
sieve = [ False ] * limit NEW_LINE for i in range ( 0 , limit ) : NEW_LINE INDENT sieve [ i ] = False NEW_LINE DEDENT
x = 1 NEW_LINE while ( x * x < limit ) : NEW_LINE INDENT y = 1 NEW_LINE while ( y * y < limit ) : NEW_LINE DEDENT
n = ( 4 * x * x ) + ( y * y ) NEW_LINE if ( n <= limit and ( n % 12 == 1 or n % 12 == 5 ) ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT n = ( 3 * x * x ) + ( y * y ) NEW_LINE if ( n <= limit and n % 12 == 7 ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT n = ( 3 * x * x ) - ( y * y ) NEW_LINE if ( x > y and n <= limit and n % 12 == 11 ) : NEW_LINE INDENT sieve [ n ] ^= True NEW_LINE DEDENT y += 1 NEW_LINE x += 1 NEW_LINE
r = 5 NEW_LINE while ( r * r < limit ) : NEW_LINE INDENT if ( sieve [ r ] ) : NEW_LINE INDENT for i in range ( r * r , limit , r * r ) : NEW_LINE INDENT sieve [ i ] = False NEW_LINE DEDENT DEDENT DEDENT
for a in range ( 5 , limit ) : NEW_LINE INDENT if ( sieve [ a ] ) : NEW_LINE INDENT print ( a , end = " ▁ " ) NEW_LINE DEDENT DEDENT
limit = 20 NEW_LINE SieveOfAtkin ( limit ) NEW_LINE
def isInside ( circle_x , circle_y , rad , x , y ) : NEW_LINE
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) : NEW_LINE INDENT return True ; NEW_LINE DEDENT else : NEW_LINE INDENT return False ; NEW_LINE DEDENT
x = 1 ; NEW_LINE y = 1 ; NEW_LINE circle_x = 0 ; NEW_LINE circle_y = 1 ; NEW_LINE rad = 2 ; NEW_LINE if ( isInside ( circle_x , circle_y , rad , x , y ) ) : NEW_LINE INDENT print ( " Inside " ) ; NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Outside " ) ; NEW_LINE DEDENT
def eval ( a , op , b ) : NEW_LINE INDENT if op == ' + ' : return a + b NEW_LINE if op == ' - ' : return a - b NEW_LINE if op == ' * ' : return a * b NEW_LINE DEDENT
def evaluateAll ( expr , low , high ) : NEW_LINE
res = [ ] NEW_LINE
if low == high : NEW_LINE INDENT res . append ( int ( expr [ low ] ) ) NEW_LINE return res NEW_LINE DEDENT
if low == ( high - 2 ) : NEW_LINE INDENT num = eval ( int ( expr [ low ] ) , expr [ low + 1 ] , int ( expr [ low + 2 ] ) ) NEW_LINE res . append ( num ) NEW_LINE return res NEW_LINE DEDENT
for i in range ( low + 1 , high + 1 , 2 ) : NEW_LINE
' NEW_LINE INDENT l = evaluateAll ( expr , low , i - 1 ) NEW_LINE DEDENT
' NEW_LINE INDENT r = evaluateAll ( expr , i + 1 , high ) NEW_LINE DEDENT
' NEW_LINE INDENT for s1 in range ( 0 , len ( l ) ) : NEW_LINE DEDENT
' NEW_LINE INDENT for s2 in range ( 0 , len ( r ) ) : NEW_LINE DEDENT
val = eval ( l [ s1 ] , expr [ i ] , r [ s2 ] ) NEW_LINE res . append ( val ) NEW_LINE return res NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT expr = "1*2 + 3*4" NEW_LINE length = len ( expr ) NEW_LINE ans = evaluateAll ( expr , 0 , length - 1 ) NEW_LINE for i in range ( 0 , len ( ans ) ) : NEW_LINE INDENT print ( ans [ i ] ) NEW_LINE DEDENT DEDENT
def isLucky ( n ) : NEW_LINE
ar = [ 0 ] * 10 NEW_LINE
while ( n > 0 ) : NEW_LINE
digit = math . floor ( n % 10 ) NEW_LINE
if ( ar [ digit ] ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
ar [ digit ] = 1 NEW_LINE
n = n / 10 NEW_LINE return 1 NEW_LINE
arr = [ 1291 , 897 , 4566 , 1232 , 80 , 700 ] NEW_LINE n = len ( arr ) NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT k = arr [ i ] NEW_LINE if ( isLucky ( k ) ) : NEW_LINE INDENT print ( k , " ▁ is ▁ Lucky ▁ " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( k , " ▁ is ▁ not ▁ Lucky ▁ " ) NEW_LINE DEDENT DEDENT
def printSquares ( n ) : NEW_LINE
square = 0 NEW_LINE odd = 1 NEW_LINE
for x in range ( 0 , n ) : NEW_LINE
print ( square , end = " ▁ " ) NEW_LINE
' NEW_LINE INDENT square = square + odd NEW_LINE odd = odd + 2 NEW_LINE DEDENT
n = 5 ; NEW_LINE printSquares ( n ) NEW_LINE
rev_num = 0 NEW_LINE base_pos = 1 NEW_LINE
def reversDigits ( num ) : NEW_LINE INDENT global rev_num NEW_LINE global base_pos NEW_LINE if ( num > 0 ) : NEW_LINE INDENT reversDigits ( ( int ) ( num / 10 ) ) NEW_LINE rev_num += ( num % 10 ) * base_pos NEW_LINE base_pos *= 10 NEW_LINE DEDENT return rev_num NEW_LINE DEDENT
num = 4562 NEW_LINE print ( " Reverse ▁ of ▁ no . ▁ is ▁ " , reversDigits ( num ) ) NEW_LINE
def printSubsets ( n ) : NEW_LINE INDENT i = n NEW_LINE while ( i != 0 ) : NEW_LINE INDENT print ( i , end = " ▁ " ) NEW_LINE i = ( i - 1 ) & n NEW_LINE DEDENT print ( "0" ) NEW_LINE DEDENT
n = 9 NEW_LINE printSubsets ( n ) NEW_LINE
def isDivisibleby17 ( n ) : NEW_LINE
if ( n == 0 or n == 17 ) : NEW_LINE INDENT return True NEW_LINE DEDENT
if ( n < 17 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) NEW_LINE
n = 35 NEW_LINE if ( isDivisibleby17 ( n ) ) : NEW_LINE INDENT print ( n , " is ▁ divisible ▁ by ▁ 17" ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( n , " is ▁ not ▁ divisible ▁ by ▁ 17" ) NEW_LINE DEDENT
def answer ( n ) : NEW_LINE
m = 2 ; NEW_LINE
ans = 1 ; NEW_LINE r = 1 ; NEW_LINE
while r < n : NEW_LINE
r = ( int ) ( ( pow ( 2 , m ) - 1 ) * ( pow ( 2 , m - 1 ) ) ) ; NEW_LINE
if r < n : NEW_LINE INDENT ans = r ; NEW_LINE DEDENT
m = m + 1 ; NEW_LINE return ans ; NEW_LINE
print ( answer ( 7 ) ) ; NEW_LINE
def setBitNumber ( n ) : NEW_LINE INDENT if ( n == 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT msb = 0 ; NEW_LINE n = int ( n / 2 ) ; NEW_LINE while ( n > 0 ) : NEW_LINE INDENT n = int ( n / 2 ) ; NEW_LINE msb += 1 ; NEW_LINE DEDENT return ( 1 << msb ) ; NEW_LINE DEDENT
n = 0 ; NEW_LINE print ( setBitNumber ( n ) ) ; NEW_LINE
def setBitNumber ( n ) : NEW_LINE
n |= n >> 1 NEW_LINE
n |= n >> 2 NEW_LINE n |= n >> 4 NEW_LINE n |= n >> 8 NEW_LINE n |= n >> 16 NEW_LINE
n = n + 1 NEW_LINE
return ( n >> 1 ) NEW_LINE
n = 273 NEW_LINE print ( setBitNumber ( n ) ) NEW_LINE
def countTrailingZero ( x ) : NEW_LINE INDENT count = 0 NEW_LINE while ( ( x & 1 ) == 0 ) : NEW_LINE INDENT x = x >> 1 NEW_LINE count += 1 NEW_LINE DEDENT return count NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( countTrailingZero ( 11 ) ) NEW_LINE DEDENT
def countTrailingZero ( x ) : NEW_LINE
lookup = [ 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 ] NEW_LINE
return lookup [ ( - x & x ) % 37 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT print ( countTrailingZero ( 48 ) ) NEW_LINE DEDENT
def multiplyBySevenByEight ( n ) : NEW_LINE
' NEW_LINE INDENT return ( n - ( n >> 3 ) ) NEW_LINE DEDENT
n = 9 NEW_LINE print ( multiplyBySevenByEight ( n ) ) NEW_LINE
def multiplyBySevenByEight ( n ) : NEW_LINE
return ( ( n << 3 ) - n ) >> 3 ; NEW_LINE
n = 15 ; NEW_LINE print ( multiplyBySevenByEight ( n ) ) ; NEW_LINE
def getMaxMedian ( arr , n , k ) : NEW_LINE INDENT size = n + k NEW_LINE DEDENT
arr . sort ( reverse = False ) NEW_LINE
if ( size % 2 == 0 ) : NEW_LINE INDENT median = ( arr [ int ( size / 2 ) - 1 ] + arr [ int ( size / 2 ) ] ) / 2 NEW_LINE return median NEW_LINE DEDENT
median = arr [ int ( size / 2 ) ] NEW_LINE return median NEW_LINE
def printSorted ( a , b , c ) : NEW_LINE
get_max = max ( a , max ( b , c ) ) NEW_LINE
get_min = - max ( - a , max ( - b , - c ) ) NEW_LINE get_mid = ( a + b + c ) - ( get_max + get_min ) NEW_LINE print ( get_min , " ▁ " , get_mid , " ▁ " , get_max ) NEW_LINE
a , b , c = 4 , 1 , 9 NEW_LINE printSorted ( a , b , c ) NEW_LINE
def insertionSort ( arr ) : NEW_LINE INDENT for i in range ( 1 , len ( arr ) ) : NEW_LINE INDENT key = arr [ i ] NEW_LINE DEDENT DEDENT
j = i - 1 NEW_LINE while j >= 0 and key < arr [ j ] : NEW_LINE INDENT arr [ j + 1 ] = arr [ j ] NEW_LINE j -= 1 NEW_LINE DEDENT arr [ j + 1 ] = key NEW_LINE
arr = [ 12 , 11 , 13 , 5 , 6 ] NEW_LINE insertionSort ( arr ) NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT print ( " % ▁ d " % arr [ i ] ) NEW_LINE DEDENT
def countPaths ( n , m ) : NEW_LINE
if ( n == 0 or m == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) NEW_LINE
n = 3 NEW_LINE m = 2 NEW_LINE print ( " Number ▁ of ▁ Paths " , countPaths ( n , m ) ) NEW_LINE
def count ( S , m , n ) : NEW_LINE
if ( n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT
if ( n < 0 ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT
if ( m <= 0 and n >= 1 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; NEW_LINE
arr = [ 1 , 2 , 3 ] NEW_LINE m = len ( arr ) NEW_LINE print ( count ( arr , m , 4 ) ) NEW_LINE
def isVowel ( c ) : NEW_LINE INDENT return ( c == ' a ' or c == ' e ' or c == ' i ' or c == ' o ' or c == ' u ' ) NEW_LINE DEDENT
def encryptString ( s , n , k ) : NEW_LINE INDENT countVowels = 0 NEW_LINE countConsonants = 0 NEW_LINE ans = " " NEW_LINE DEDENT
for l in range ( n - k + 1 ) : NEW_LINE INDENT countVowels = 0 NEW_LINE countConsonants = 0 NEW_LINE DEDENT
for r in range ( l , l + k ) : NEW_LINE
if ( isVowel ( s [ r ] ) == True ) : NEW_LINE INDENT countVowels += 1 NEW_LINE DEDENT else : NEW_LINE INDENT countConsonants += 1 NEW_LINE DEDENT
ans += ( str ) ( countVowels * countConsonants ) NEW_LINE return ans NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT s = " hello " NEW_LINE n = len ( s ) NEW_LINE k = 2 NEW_LINE print ( encryptString ( s , n , k ) ) NEW_LINE DEDENT
def findVolume ( a ) : NEW_LINE
if ( a < 0 ) : NEW_LINE INDENT return - 1 NEW_LINE DEDENT
r = a / 2 NEW_LINE
h = a NEW_LINE
V = 3.14 * pow ( r , 2 ) * h NEW_LINE return V NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT a = 5 NEW_LINE print ( findVolume ( a ) ) NEW_LINE DEDENT
def volumeTriangular ( a , b , h ) : NEW_LINE INDENT return ( 0.1666 ) * a * b * h NEW_LINE DEDENT
def volumeSquare ( b , h ) : NEW_LINE INDENT return ( 0.33 ) * b * b * h NEW_LINE DEDENT
def volumePentagonal ( a , b , h ) : NEW_LINE INDENT return ( 0.83 ) * a * b * h NEW_LINE DEDENT
def volumeHexagonal ( a , b , h ) : NEW_LINE INDENT return a * b * h NEW_LINE DEDENT
b = float ( 4 ) NEW_LINE h = float ( 9 ) NEW_LINE a = float ( 4 ) NEW_LINE print ( " Volume ▁ of ▁ triangular ▁ base ▁ pyramid ▁ is ▁ " , volumeTriangular ( a , b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ square ▁ base ▁ pyramid ▁ is ▁ " , volumeSquare ( b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ pentagonal ▁ base ▁ pyramid ▁ is ▁ " , volumePentagonal ( a , b , h ) ) NEW_LINE print ( " Volume ▁ of ▁ Hexagonal ▁ base ▁ pyramid ▁ is ▁ " , volumeHexagonal ( a , b , h ) ) NEW_LINE
def Area ( b1 , b2 , h ) : NEW_LINE INDENT return ( ( b1 + b2 ) / 2 ) * h NEW_LINE DEDENT
base1 = 8 ; base2 = 10 ; height = 6 NEW_LINE area = Area ( base1 , base2 , height ) NEW_LINE print ( " Area ▁ is : " , area ) NEW_LINE
def numberOfDiagonals ( n ) : NEW_LINE INDENT return n * ( n - 3 ) / 2 NEW_LINE DEDENT
def main ( ) : NEW_LINE INDENT n = 5 NEW_LINE print ( n , " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ) NEW_LINE print ( numberOfDiagonals ( n ) , " ▁ diagonals " ) NEW_LINE DEDENT if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT main ( ) NEW_LINE DEDENT
def Printksubstring ( str1 , n , k ) : NEW_LINE
total = int ( ( n * ( n + 1 ) ) / 2 ) NEW_LINE
if ( k > total ) : NEW_LINE INDENT print ( " - 1" ) NEW_LINE return NEW_LINE DEDENT
substring = [ 0 for i in range ( n + 1 ) ] NEW_LINE substring [ 0 ] = 0 NEW_LINE
temp = n NEW_LINE for i in range ( 1 , n + 1 , 1 ) : NEW_LINE
substring [ i ] = substring [ i - 1 ] + temp NEW_LINE temp -= 1 NEW_LINE
l = 1 NEW_LINE h = n NEW_LINE start = 0 NEW_LINE while ( l <= h ) : NEW_LINE INDENT m = int ( ( l + h ) / 2 ) NEW_LINE if ( substring [ m ] > k ) : NEW_LINE INDENT start = m NEW_LINE h = m - 1 NEW_LINE DEDENT elif ( substring [ m ] < k ) : NEW_LINE INDENT l = m + 1 NEW_LINE DEDENT else : NEW_LINE INDENT start = m NEW_LINE break NEW_LINE DEDENT DEDENT
end = n - ( substring [ start ] - k ) NEW_LINE
for i in range ( start - 1 , end ) : NEW_LINE INDENT print ( str1 [ i ] , end = " " ) NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT str1 = " abc " NEW_LINE k = 4 NEW_LINE n = len ( str1 ) NEW_LINE Printksubstring ( str1 , n , k ) NEW_LINE DEDENT
def LowerInsertionPoint ( arr , n , X ) : NEW_LINE
if ( X < arr [ 0 ] ) : NEW_LINE INDENT return 0 ; NEW_LINE DEDENT elif ( X > arr [ n - 1 ] ) : NEW_LINE INDENT return n NEW_LINE DEDENT lowerPnt = 0 NEW_LINE i = 1 NEW_LINE while ( i < n and arr [ i ] < X ) : NEW_LINE INDENT lowerPnt = i NEW_LINE i = i * 2 NEW_LINE DEDENT
while ( lowerPnt < n and arr [ lowerPnt ] < X ) : NEW_LINE INDENT lowerPnt += 1 NEW_LINE DEDENT return lowerPnt NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE X = 4 NEW_LINE print ( LowerInsertionPoint ( arr , n , X ) ) NEW_LINE DEDENT
def getCount ( M , N ) : NEW_LINE INDENT count = 0 ; NEW_LINE DEDENT
if ( M == 1 ) : NEW_LINE INDENT return N ; NEW_LINE DEDENT
if ( N == 1 ) : NEW_LINE INDENT return M ; NEW_LINE DEDENT if ( N > M ) : NEW_LINE
for i in range ( 1 , M + 1 ) : NEW_LINE INDENT numerator = N * i - N + M - i ; NEW_LINE denominator = M - 1 ; NEW_LINE DEDENT
if ( numerator % denominator == 0 ) : NEW_LINE INDENT j = numerator / denominator ; NEW_LINE DEDENT
if ( j >= 1 and j <= N ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT else : NEW_LINE
for j in range ( 1 , N + 1 ) : NEW_LINE INDENT numerator = M * j - M + N - j ; NEW_LINE denominator = N - 1 ; NEW_LINE DEDENT
if ( numerator % denominator == 0 ) : NEW_LINE INDENT i = numerator / denominator ; NEW_LINE DEDENT
if ( i >= 1 and i <= M ) : NEW_LINE INDENT count += 1 ; NEW_LINE DEDENT return count ; NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT M , N = 3 , 5 ; NEW_LINE print ( getCount ( M , N ) ) ; NEW_LINE DEDENT
def middleOfThree ( a , b , c ) : NEW_LINE
if a > b : NEW_LINE INDENT if ( b > c ) : NEW_LINE INDENT return b NEW_LINE DEDENT elif ( a > c ) : NEW_LINE INDENT return c NEW_LINE DEDENT else : NEW_LINE INDENT return a NEW_LINE DEDENT DEDENT else : NEW_LINE
if ( a > c ) : NEW_LINE INDENT return a NEW_LINE DEDENT elif ( b > c ) : NEW_LINE INDENT return c NEW_LINE DEDENT else : NEW_LINE INDENT return b NEW_LINE DEDENT
a = 20 NEW_LINE b = 30 NEW_LINE c = 40 NEW_LINE print ( middleOfThree ( a , b , c ) ) NEW_LINE
def printArr ( arr , n ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT print ( arr [ i ] , end = " " ) NEW_LINE DEDENT DEDENT
def compare ( num1 , num2 ) : NEW_LINE
A = str ( num1 ) NEW_LINE
B = str ( num2 ) NEW_LINE
return int ( A + B ) <= int ( B + A ) NEW_LINE
def sort ( arr ) : NEW_LINE INDENT for i in range ( len ( arr ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( arr ) ) : NEW_LINE INDENT if compare ( arr [ i ] , arr [ j ] ) == False : NEW_LINE INDENT arr [ i ] , arr [ j ] = arr [ j ] , arr [ i ] NEW_LINE DEDENT DEDENT DEDENT DEDENT
def printSmallest ( N , arr ) : NEW_LINE
sort ( arr ) NEW_LINE
printArr ( arr , N ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 5 , 6 , 2 , 9 , 21 , 1 ] NEW_LINE N = len ( arr ) NEW_LINE printSmallest ( N , arr ) NEW_LINE DEDENT
def isPossible ( a , b , n , k ) : NEW_LINE
a . sort ( reverse = True ) NEW_LINE
b . sort ( ) NEW_LINE
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] + b [ i ] < k ) : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT return True NEW_LINE
a = [ 2 , 1 , 3 ] NEW_LINE b = [ 7 , 8 , 9 ] NEW_LINE k = 10 NEW_LINE n = len ( a ) NEW_LINE if ( isPossible ( a , b , n , k ) ) : NEW_LINE INDENT print ( " Yes " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No " ) NEW_LINE DEDENT
def encryptString ( string , n ) : NEW_LINE INDENT i , cnt = 0 , 0 NEW_LINE encryptedStr = " " NEW_LINE while i < n : NEW_LINE DEDENT
cnt = i + 1 NEW_LINE
while cnt > 0 : NEW_LINE INDENT encryptedStr += string [ i ] NEW_LINE cnt -= 1 NEW_LINE DEDENT i += 1 NEW_LINE return encryptedStr NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " geeks " NEW_LINE n = len ( string ) NEW_LINE print ( encryptString ( string , n ) ) NEW_LINE DEDENT
def minDiff ( n , x , A ) : NEW_LINE INDENT mn = A [ 0 ] NEW_LINE mx = A [ 0 ] NEW_LINE DEDENT
for i in range ( 0 , n ) : NEW_LINE INDENT mn = min ( mn , A [ i ] ) NEW_LINE mx = max ( mx , A [ i ] ) NEW_LINE DEDENT
return max ( 0 , mx - mn - 2 * x ) NEW_LINE
n = 3 NEW_LINE x = 3 NEW_LINE A = [ 1 , 3 , 6 ] NEW_LINE
print ( minDiff ( n , x , A ) ) NEW_LINE
