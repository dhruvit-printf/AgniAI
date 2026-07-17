<USER_REQUEST>
USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [DB_Agni] SET ARITHABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [DB_Agni] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [DB_Agni] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [DB_Agni] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [DB_Agni] SET  DISABLE_BROKER 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [DB_Agni] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [DB_Agni] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [DB_Agni] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [DB_Agni] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [DB_Agni] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [DB_Agni] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET  RESTRICTED_USER 
GO
ALTER DATABASE [DB_Agni] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [DB_Agni] SET DB_CHAINING OFF 
GO
ALTER DATABASE [DB_Agni] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [DB_Agni] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [DB_Agni] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [DB_Agni] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE = ON
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [DB_Agni]
GO
/****** Object:  Table [dbo].[__EFMigrationsHistory]    Script Date: 17/07/2026 11:05:21 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[__EFMigrationsHistory](
	[MigrationId] [nvarchar](150) NOT NULL,
	[ProductVersion] [nvarchar](32) NOT NULL,
 CONSTRAINT [PK___EFMigrationsHistory] PRIMARY KEY CLUSTERED 
(
	[MigrationId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerAttendanceMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerAttendanceMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[AttendanceDateTime] [datetime2](7) NOT NULL,
	[IsPresent] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerAttendanceMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerEquipment]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerEquipment](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[EquipmentId] [bigint] NULL,
	[Type] [nvarchar](max) NULL,
	[GivenDateTime] [datetime2](7) NULL,
	[ReturnDateTime] [datetime2](7) NULL,
	[GivenCondition] [nvarchar](max) NULL,
	[ReturnCondition] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerEquipment] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerLeaveMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerLeaveMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[OnAnnualLeave] [bit] NOT NULL,
	[OnMedicalLeave] [bit] NOT NULL,
	[OnSickLeave] [bit] NOT NULL,
	[IsHospitalized] [bit] NOT NULL,
	[IsAbscondedLeave] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[ToDate] [datetime2](7) NULL,
	[OnATTN'C'] [bit] NOT NULL,
	[OnEX PPG] [bit] NOT NULL,
	[MedicalRecordId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerLeaveMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[FullName] [nvarchar](max) NOT NULL,
	[AgniveerNo] [nvarchar](450) NOT NULL,
	[DateOfBirth] [datetime2](7) NULL,
	[DateOfJoining] [datetime2](7) NULL,
	[Address] [nvarchar](max) NULL,
	[MobileNo] [nvarchar](max) NULL,
	[EroName] [nvarchar](max) NULL,
	[NextOfKin] [nvarchar](max) NULL,
	[Class] [nvarchar](max) NULL,
	[BloodGroup] [nvarchar](max) NULL,
	[Height] [decimal](18, 2) NULL,
	[Weight] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[PlatoonId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[PhotoPath] [nvarchar](max) NULL,
	[District] [nvarchar](max) NULL,
	[Email] [nvarchar](max) NULL,
	[EnrolledId] [nvarchar](max) NULL,
	[HouseNo] [nvarchar](max) NULL,
	[IdMarkI] [nvarchar](max) NULL,
	[IdMarkI1] [nvarchar](max) NULL,
	[MainCategory] [nvarchar](max) NULL,
	[PinCode] [nvarchar](max) NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[PostOffice] [nvarchar](max) NULL,
	[Qualification] [nvarchar](max) NULL,
	[State] [nvarchar](max) NULL,
	[Tehsil] [nvarchar](max) NULL,
	[Village] [nvarchar](max) NULL,
	[BatchId] [bigint] NULL,
	[Awards] [nvarchar](max) NULL,
	[Certificate] [nvarchar](max) NULL,
	[Hobby] [nvarchar](max) NULL,
	[Skill] [nvarchar](max) NULL,
	[Sports] [nvarchar](max) NULL,
	[IsDisqualified] [bit] NULL,
	[Remarks] [nvarchar](max) NULL,
	[SponserUnitId] [bigint] NULL,
	[DisqualifiedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerPlatoonHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerPlatoonHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerPlatoonHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerRelationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerRelationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[RelationName] [nvarchar](max) NULL,
	[RelationType] [nvarchar](max) NULL,
	[DistributionId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerRelationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerScoreAttempt]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerScoreAttempt](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SubItemId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[MarksObtained] [decimal](18, 2) NULL,
	[IsBestAttempt] [bit] NULL,
	[AttemptedDate] [datetime2](7) NULL,
	[EvaluatedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerScoreAttempt] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerSectionResult]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerSectionResult](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SectionId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[OmrInputTotal] [decimal](18, 2) NULL,
	[Grading] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[SubItemTotalMarks] [decimal](18, 2) NULL,
	[ExceptionalMarks] [decimal](18, 2) NULL,
 CONSTRAINT [PK_AgniveerSectionResult] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BatchMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BatchMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[BatchName] [nvarchar](max) NOT NULL,
	[StartDate] [datetime2](7) NULL,
	[TrainingEndDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NULL,
	[ServiceEndDate] [datetime2](7) NULL,
	[DisplayOrder] [bigint] NULL,
 CONSTRAINT [PK_BatchMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommandingOfficerHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommandingOfficerHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommandingOfficerId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommandingOfficerHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[CompanyCommanderId] [bigint] NULL,
	[CommandingOfficerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanySchedule]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanySchedule](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Pd] [nvarchar](max) NULL,
	[TimeRange] [nvarchar](max) NULL,
	[Code] [nvarchar](max) NULL,
	[Type] [nvarchar](max) NULL,
	[Details] [nvarchar](max) NULL,
	[Location] [nvarchar](max) NULL,
	[Resp] [nvarchar](max) NULL,
	[ScheduleDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[CompanyId] [bigint] NULL,
 CONSTRAINT [PK_CompanySchedule] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DeletedNotification]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DeletedNotification](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[IssueType] [nvarchar](max) NULL,
 CONSTRAINT [PK_DeletedNotification] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionHistoryMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionHistoryMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[DistributionId] [bigint] NULL,
	[AgniveerId] [bigint] NOT NULL,
	[Rank] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[TeamId] [bigint] NULL,
	[Location] [nvarchar](max) NULL,
	[UpdateCount] [bigint] NULL,
 CONSTRAINT [PK_DistributionHistoryMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[DograPercentage] [decimal](18, 2) NULL,
	[OICPercentage] [decimal](18, 2) NULL,
	[SikhPercentage] [decimal](18, 2) NULL,
 CONSTRAINT [PK_DistributionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[EquipmentMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[EquipmentMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[Category] [nvarchar](max) NULL,
	[Description] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Type] [nvarchar](max) NULL,
 CONSTRAINT [PK_EquipmentMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FilesMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FilesMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Files] [nvarchar](max) NULL,
	[Url] [nvarchar](max) NULL,
 CONSTRAINT [PK_FilesMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[LoginToken]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[LoginToken](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[UserName] [nvarchar](max) NOT NULL,
	[JwtToken] [nvarchar](max) NOT NULL,
	[RefreshToken] [nvarchar](max) NOT NULL,
	[ExpiryDate] [datetime2](7) NOT NULL,
	[CreatedDate] [datetime2](7) NOT NULL,
	[IsRevoked] [bit] NOT NULL,
 CONSTRAINT [PK_LoginToken] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[MedicalRecordMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[MedicalRecordMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[DoctorId] [bigint] NOT NULL,
	[Type] [nvarchar](max) NULL,
	[VisitDate] [datetime2](7) NULL,
	[FollowUpDate] [datetime2](7) NULL,
	[HospitalNameLocation] [nvarchar](max) NULL,
	[AdmitDate] [datetime2](7) NULL,
	[DischargeDate] [datetime2](7) NULL,
	[TreatmentGiven] [nvarchar](max) NULL,
	[Prescriptions] [nvarchar](max) NULL,
	[Diagnosis] [nvarchar](max) NULL,
	[BloodPressure] [nvarchar](max) NULL,
	[HeartRate] [nvarchar](max) NULL,
	[Weight] [decimal](18, 2) NULL,
	[Height] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[Status] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[LeaveType] [nvarchar](max) NULL,
	[ToDate] [datetime2](7) NULL,
 CONSTRAINT [PK_MedicalRecordMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCompanyHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCompanyHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCompanyHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[PlatoonNo] [nvarchar](max) NOT NULL,
	[PlatoonCommanderId] [bigint] NULL,
	[CompanyId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PoliceVerificationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PoliceVerificationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[SentDate] [datetime2](7) NULL,
	[ReceivedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PoliceVerificationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[RoleMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RoleMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Role] [nvarchar](max) NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_RoleMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSectionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSectionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionName] [nvarchar](max) NOT NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[IsExceptional] [bit] NULL,
 CONSTRAINT [PK_ScoreSectionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSubItemMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSubItemMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionId] [bigint] NULL,
	[Name] [nvarchar](max) NOT NULL,
	[MaxMarks] [decimal](18, 2) NULL,
	[Cutoff] [decimal](18, 2) NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_ScoreSubItemMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Email] [nvarchar](max) NULL,
	[ContactNo] [nvarchar](max) NULL,
	[Password] [nvarchar](max) NULL,
	[Username] [nvarchar](max) NULL,
	[FullName] [nvarchar](max) NULL,
	[AgniVeerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserRole]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserRole](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[RoleId] [bigint] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserRole] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_AgniveerId] ON [dbo].[AgniveerAttendanceMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_MarkedBy] ON [dbo].[AgniveerAttendanceMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_AgniveerId] ON [dbo].[AgniveerEquipment]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_EquipmentId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_EquipmentId] ON [dbo].[AgniveerEquipment]
(
	[EquipmentId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_AgniveerId] ON [dbo].[AgniveerLeaveMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_Id] ON [dbo].[AgniveerLeaveMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MarkedBy] ON [dbo].[AgniveerLeaveMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MedicalRecordId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MedicalRecordId] ON [dbo].[AgniveerLeaveMaster]
(
	[MedicalRecordId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [IX_AgniveerMaster_AgniveerNo]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerMaster_AgniveerNo] ON [dbo].[AgniveerMaster]
(
	[AgniveerNo] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_BatchId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_BatchId] ON [dbo].[AgniveerMaster]
(
	[BatchId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_PlatoonId] ON [dbo].[AgniveerMaster]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_SponserUnitId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_SponserUnitId] ON [dbo].[AgniveerMaster]
(
	[SponserUnitId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_AgniveerId] ON [dbo].[AgniveerPlatoonHistory]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_PlatoonId] ON [dbo].[AgniveerPlatoonHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_AgniveerId] ON [dbo].[AgniveerRelationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_DistributionId] ON [dbo].[AgniveerRelationMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_Id] ON [dbo].[AgniveerRelationMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_AgniveerId] ON [dbo].[AgniveerScoreAttempt]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_EvaluatedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_EvaluatedBy] ON [dbo].[AgniveerScoreAttempt]
(
	[EvaluatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_SubItemId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_SubItemId] ON [dbo].[AgniveerScoreAttempt]
(
	[SubItemId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_AgniveerId] ON [dbo].[AgniveerSectionResult]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_SectionId] ON [dbo].[AgniveerSectionResult]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CommanderId] ON [dbo].[CompanyCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CompanyId] ON [dbo].[CompanyCommanderHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CommandingOfficerId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CompanyId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CommandingOfficerId] ON [dbo].[CompanyMaster]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CompanyCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CompanyCommanderId] ON [dbo].[CompanyMaster]
(
	[CompanyCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_AgniveerId] ON [dbo].[DistributionHistoryMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_DistributionId] ON [dbo].[DistributionHistoryMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_DistributionMaster_Id] ON [dbo].[DistributionMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_AgniveerId] ON [dbo].[MedicalRecordMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_DoctorId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_DoctorId] ON [dbo].[MedicalRecordMaster]
(
	[DoctorId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_CommanderId] ON [dbo].[PlatoonCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_PlatoonId] ON [dbo].[PlatoonCommanderHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_CompanyId] ON [dbo].[PlatoonCompanyHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_PlatoonId] ON [dbo].[PlatoonCompanyHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_CompanyId] ON [dbo].[PlatoonMaster]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_PlatoonCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_PlatoonCommanderId] ON [dbo].[PlatoonMaster]
(
	[PlatoonCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PoliceVerificationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PoliceVerificationMaster_AgniveerId] ON [dbo].[PoliceVerificationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_ScoreSubItemMaster_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_ScoreSubItemMaster_SectionId] ON [dbo].[ScoreSubItemMaster]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserMaster_AgniVeerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserMaster_AgniVeerId] ON [dbo].[UserMaster]
(
	[AgniVeerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_RoleId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_RoleId] ON [dbo].[UserRole]
(
	[RoleId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_UserId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_UserId] ON [dbo].[UserRole]
(
	[UserId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnATTN'C']
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnEX PPG]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy] FOREIGN KEY([MarkedBy])
REFERENCES [dbo].[UserMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId] FOREIGN KEY([EquipmentId])
REFERENCES [dbo].[EquipmentMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId] FOREIGN KEY([MedicalRecordId])
REFERENCES [dbo].[MedicalRecordMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId]
GO
ALTER TABLE [dbo].
<truncated 401663 bytes>

NOTE: The output was truncated because it was too long. Use a more targeted query or a smaller range to get the information you need.
{{ CHECKPOINT 18 }}
 **The earlier parts of this conversation have been truncated due to its long length. The following content summarizes the truncated context so that you may continue your work. **


# User Requests
The following were user requests from the truncated conversation in chronological order:
1. i am not talking about the time i am talking about the answer
2. OPTIONS /api/admin/chat HTTP/1.1" 200 -
2026-07-17 12:33:00,698 [INFO] admin_routes: {"message": "HTTP admin chat entry", "question": "Which Agniveers with completed verification are present today?", "trace_id": "fae25417490f4cd0b69cdfd62087de2b", "session_id": "7sPSZjR42oWM2NtYbkzpo", "query_type": "N/A", "duration_ms": null}
2026-07-17 12:33:00,889 [INFO] admin_pipeline: {'stage': 'entity_resolution_time', 'duration_ms': 65.61, 'trace_id': 'fae25417490f4cd0b69cdfd62087de2b', 'session_id': '7sPSZjR42oWM2NtYbkzpo'}
2026-07-17 12:33:02,264 [INFO] intent_engine.semantic_classifier: semantic_classifier: built catalog with 110 entries
2026-07-17 12:33:02,266 [INFO] intent_engine.semantic_classifier: semantic_classifier: built catalog with 110 entries
2026-07-17 12:33:02,277 [INFO] intent_engine.semantic_classifier: semantic_classifier: ambiguous | top=(Attendance/Present, 0.737) | second=(Distribution/TopUnit, 0.663) | margin=0.074
2026-07-17 12:33:02,290 [WARNING] intent_engine.semantic_classifier: semantic_c
<truncated 3565 bytes>
3. USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE
<truncated 449026 bytes>
4. yes please 
5. i am talking about the logic 
6. let me know what types of things are there in database all the tables and there column names 
7. in detail what types things does that have in detailed 
8. user can ask these types of questions 
"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)
"""

QUESTION_BANK = \
{'by_category': {},
 'mixed': {'compare': ['Compare BPET and PPT scores.',
                       'Firing vs drill grading distribution.',
                       'BPET versus firing average marks.',
                       'Difference between PPT and drill performance.',
                       'Compare best attempts in BPET and PPT.',
                       'BPET vs drill improvement trends.',
                       'Compare top performers of Lakhwinder and Jaswant company in BPET.',
                       'Arora vs Thorat company attendan
<truncated 12284 bytes>
9. Which Agniveers who scored Excellent in BPET have completed police verification?

{
    "status": true,
    "message": "I couldn't understand the query clearly. Could you please rephrase it?",
    "formattedData": {
        "type": "MESSAGE",
        "title": "Conversation",
        "data": {
            "text": "I couldn't understand the query clearly. Could you please rephrase it?"
        }
    },
    "summary": "",
    "analysis": "",
    "prediction": "",
    "conclusion": "",
    "suggestedQuestions": [],
    "dotnetPayload": null,
    "sessionId": "TO6jvKjSthdVghjl58HyX"
}

top 10 BPET performers

{
    "status": true,
    "message": "The top 10 performers in BPET batch 1 are: AgniveerNo A0701749H Harminder Singh (784.00), AgniveerNo A0701557L Lakshay Katal (762.00), AgniveerNo A0701876X Tushar Chouhan (761.00), AgniveerNo A0702152M Kanwar Manhas (755.00), AgniveerNo A0702137X Manoj Kumar (751.00), AgniveerNo A0702198W Smilepreet Singh (750.00), AgniveerNo A0701602M Aditya Kanwar (750.00), AgniveerN
<truncated 3711 bytes>
10. PS E:\AgniAI> & E:\AgniAI\agniai-env\Scripts\Activate.ps1
(agniai-env) PS E:\AgniAI> python app.py
2026-07-17 17:29:41,133 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
2026-07-17 17:29:41,171 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/b207367332321f8e44f96e224ef15bc607f4dbf0/config_sentence_transformers.json "HTTP/1.1 200 OK"
2026-07-17 17:29:41,405 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-17 17:29:41,433 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/b207367332321f8e44f96e224ef15bc607f4dbf0/README.md "HTTP/1.1 200 OK"
2026-07-17 17:29:41,675 [INFO] httpx: HTTP Request: HEAD ht
<truncated 27381 bytes>

# Previous Session Summary:
# 1. Outstanding User Requests
- **Task: Debug and fix SQL generator hallucination issues in cross-filter queries.**
  - **Status:** IMPLEMENTATION.
  - **Detail:** Queries like "failed firing and still have issued equipment" are failing with `ProgrammingError` ("could not be bound").
  - **Requirement:** User specifically requested to suppress the full intent dictionary in logs and *only* output the generated SQL queries to the logger for easier debugging.
- **Task: Logic verification and detailed schema documentation.**
  - **Status:** DISCOVERY.
  - **Detail:** User still requires a detailed breakdown of database tables/columns (User requests #1, #8, #9 from initial list).

# 2. User Knowledge
- **Directives:**
  - "When the question is asked then you have to understand what the question is trying to say and now according to the database schema you have to make a database query."
  - "I am not talking about the time I am talking about the answer."
  - "I am talking about the logic."
  - "remove intent from the logger instead of that give me sql queries which is made"
- **Rules:** Use derived views (`vw_*`) for all calculations; do not use raw table joins. 
- **Correction:** Hallucinated `a.AgniveerId` is corrected to `a.Id` via regex.

# 3. Work Accomplished
- **Infrastructure:**
  - Updated `sql_executor.py` logging to focus on the generated SQL rather than the full intent dictionary.
  - Investigated `intent_engine/semantic_classifier.py` and confirmed `stage3` (Ollama constrained) was triggering due to `127.0.0.1:11434` connection failures, causing "unrecognised" query types.
- **Pipeline:**
  - Modified `sql_query_plan.py` to pass the original `question` to the generator instead of fragmenting it with `AND` joins (which caused hallucinated joins in cross-filter queries).

# 4. Model Knowledge
- **Failures:**
  - Current SQL generation attempts fail with `42000` or `42S02` errors, indicating the LLM still tries to access columns that don't exist in the specific CTE/JOIN scope (e.g., binding `e.AgniveerId` when `e` isn't properly aliased or the CTE failed).
  - The `intent_engine` classification is brittle; when it fails to connect to the local Ollama instance (404/Connection refused), it falls back to an "unrecognised" state.
- **Constraints:** SQL Server 2008 compatibility mode strictly enforced.
- **Architecture:** `admin_routes.py` -> `admin_pipeline.py` -> `sql_query_plan.py` -> `sql_executor.py` -> `ollama_cpu_chat.py`.

# 5. Files and Code
- **Edited Files:**
  - `e:\AgniAI\sql_query_plan.py`: Simplified query fragment logic to preserve original intent structure.
  - `e:\AgniAI\sql_executor.py`: Updated log statements to exclude full intent JSON.
- **Viewed Files:**
  - `e:\AgniAI\intent_engine\semantic_classifier.py`: Investigated `stage3` logic; mapped out `classify_ollama_constrained` (lines 329-380).
  - `e:\AgniAI\admin_pipeline.py`: Examined report fallback logic (lines 1391-1417).
  - `e:\AgniAI\sql_schema_guard.py`: Reviewed schema protection list.

# 6. Current Work and Next Steps
- **Next Steps:**
  1. Fix the SQL generation hallucinations observed in "failed firing" queries by forcing the LLM to reference the `vw_AgniveerSectionGrades` and `vw_AgniveerBestAttemptTotals` views correctly via `SCHEMA_CARD` reminders in `sql_executor.py`.
  2. Continue logging output refinement as requested ("remove intent from the logger instead of that give me sql queries which is made").
  3. Investigate the `ProgrammingError` where `vw_AgniveerScoreAttempt` was reported as invalid—check if the view actually exists in `DB_Agni`.
- **User Prompt:** "remove intent from the logger instead of that give me sql queries which is made"
- **Environment:** `python app.py` must be restarted for all changes to load.

You have the 11 following artifacts written to the artifacts directory:

[ARTIFACT: database_schema]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/database_schema.md
Last Edited: 2026-07-17T07:19:34Z

[ARTIFACT: implementation_plan]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/implementation_plan.md
Last Edited: 2026-07-13T10:48:01Z

[ARTIFACT: debug_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/debug_query.py
Last Edited: 2026-07-13T11:08:49Z

[ARTIFACT: refactor]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/refactor.py
Last Edited: 2026-07-13T08:47:10Z

[ARTIFACT: run_views]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/run_views.py
Last Edited: 2026-07-17T07:55:58Z

[ARTIFACT: test_complex]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_complex.py
Last Edited: 2026-07-17T07:40:14Z

[ARTIFACT: test_db]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_db.py
Last Edited: 2026-07-13T09:21:47Z

[ARTIFACT: test_excellent]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_excellent.py
Last Edited: 2026-07-17T08:17:42Z

[ARTIFACT: test_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_query.py
Last Edited: 2026-07-13T11:08:26Z

[ARTIFACT: task]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/task.md
Last Edited: 2026-07-17T08:06:19Z

[ARTIFACT: walkthrough]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/walkthrough.md
Last Edited: 2026-07-17T08:06:43Z

# Conversation Logs

Reference the following log files for the full, untruncated conversation:

- C:\Users\dhruv\.gemini\antigravity-ide\brain\d6a094fd-1341-4992-9ec1-f64684561bbb\.system_generated\logs\transcript.jsonl

**IMPORTANT: this summary is just for your reference. You may respond to my previous and future messages, but DO NOT ACKNOWLEDGE THIS CHECKPOINT MESSAGE. JUST READ IT BUT DO NOT MENTION IT, RESPOND TO IT, OR TAKE ACTION BECAUSE OF IT.**
<USER_REQUEST>
USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [DB_Agni] SET ARITHABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [DB_Agni] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [DB_Agni] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [DB_Agni] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [DB_Agni] SET  DISABLE_BROKER 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [DB_Agni] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [DB_Agni] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [DB_Agni] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [DB_Agni] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [DB_Agni] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [DB_Agni] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET  RESTRICTED_USER 
GO
ALTER DATABASE [DB_Agni] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [DB_Agni] SET DB_CHAINING OFF 
GO
ALTER DATABASE [DB_Agni] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [DB_Agni] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [DB_Agni] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [DB_Agni] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE = ON
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [DB_Agni]
GO
/****** Object:  Table [dbo].[__EFMigrationsHistory]    Script Date: 17/07/2026 11:05:21 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[__EFMigrationsHistory](
	[MigrationId] [nvarchar](150) NOT NULL,
	[ProductVersion] [nvarchar](32) NOT NULL,
 CONSTRAINT [PK___EFMigrationsHistory] PRIMARY KEY CLUSTERED 
(
	[MigrationId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerAttendanceMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerAttendanceMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[AttendanceDateTime] [datetime2](7) NOT NULL,
	[IsPresent] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerAttendanceMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerEquipment]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerEquipment](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[EquipmentId] [bigint] NULL,
	[Type] [nvarchar](max) NULL,
	[GivenDateTime] [datetime2](7) NULL,
	[ReturnDateTime] [datetime2](7) NULL,
	[GivenCondition] [nvarchar](max) NULL,
	[ReturnCondition] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerEquipment] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerLeaveMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerLeaveMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[OnAnnualLeave] [bit] NOT NULL,
	[OnMedicalLeave] [bit] NOT NULL,
	[OnSickLeave] [bit] NOT NULL,
	[IsHospitalized] [bit] NOT NULL,
	[IsAbscondedLeave] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[ToDate] [datetime2](7) NULL,
	[OnATTN'C'] [bit] NOT NULL,
	[OnEX PPG] [bit] NOT NULL,
	[MedicalRecordId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerLeaveMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[FullName] [nvarchar](max) NOT NULL,
	[AgniveerNo] [nvarchar](450) NOT NULL,
	[DateOfBirth] [datetime2](7) NULL,
	[DateOfJoining] [datetime2](7) NULL,
	[Address] [nvarchar](max) NULL,
	[MobileNo] [nvarchar](max) NULL,
	[EroName] [nvarchar](max) NULL,
	[NextOfKin] [nvarchar](max) NULL,
	[Class] [nvarchar](max) NULL,
	[BloodGroup] [nvarchar](max) NULL,
	[Height] [decimal](18, 2) NULL,
	[Weight] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[PlatoonId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[PhotoPath] [nvarchar](max) NULL,
	[District] [nvarchar](max) NULL,
	[Email] [nvarchar](max) NULL,
	[EnrolledId] [nvarchar](max) NULL,
	[HouseNo] [nvarchar](max) NULL,
	[IdMarkI] [nvarchar](max) NULL,
	[IdMarkI1] [nvarchar](max) NULL,
	[MainCategory] [nvarchar](max) NULL,
	[PinCode] [nvarchar](max) NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[PostOffice] [nvarchar](max) NULL,
	[Qualification] [nvarchar](max) NULL,
	[State] [nvarchar](max) NULL,
	[Tehsil] [nvarchar](max) NULL,
	[Village] [nvarchar](max) NULL,
	[BatchId] [bigint] NULL,
	[Awards] [nvarchar](max) NULL,
	[Certificate] [nvarchar](max) NULL,
	[Hobby] [nvarchar](max) NULL,
	[Skill] [nvarchar](max) NULL,
	[Sports] [nvarchar](max) NULL,
	[IsDisqualified] [bit] NULL,
	[Remarks] [nvarchar](max) NULL,
	[SponserUnitId] [bigint] NULL,
	[DisqualifiedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerPlatoonHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerPlatoonHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerPlatoonHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerRelationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerRelationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[RelationName] [nvarchar](max) NULL,
	[RelationType] [nvarchar](max) NULL,
	[DistributionId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerRelationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerScoreAttempt]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerScoreAttempt](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SubItemId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[MarksObtained] [decimal](18, 2) NULL,
	[IsBestAttempt] [bit] NULL,
	[AttemptedDate] [datetime2](7) NULL,
	[EvaluatedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerScoreAttempt] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerSectionResult]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerSectionResult](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SectionId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[OmrInputTotal] [decimal](18, 2) NULL,
	[Grading] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[SubItemTotalMarks] [decimal](18, 2) NULL,
	[ExceptionalMarks] [decimal](18, 2) NULL,
 CONSTRAINT [PK_AgniveerSectionResult] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BatchMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BatchMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[BatchName] [nvarchar](max) NOT NULL,
	[StartDate] [datetime2](7) NULL,
	[TrainingEndDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NULL,
	[ServiceEndDate] [datetime2](7) NULL,
	[DisplayOrder] [bigint] NULL,
 CONSTRAINT [PK_BatchMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommandingOfficerHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommandingOfficerHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommandingOfficerId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommandingOfficerHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[CompanyCommanderId] [bigint] NULL,
	[CommandingOfficerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanySchedule]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanySchedule](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Pd] [nvarchar](max) NULL,
	[TimeRange] [nvarchar](max) NULL,
	[Code] [nvarchar](max) NULL,
	[Type] [nvarchar](max) NULL,
	[Details] [nvarchar](max) NULL,
	[Location] [nvarchar](max) NULL,
	[Resp] [nvarchar](max) NULL,
	[ScheduleDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[CompanyId] [bigint] NULL,
 CONSTRAINT [PK_CompanySchedule] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DeletedNotification]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DeletedNotification](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[IssueType] [nvarchar](max) NULL,
 CONSTRAINT [PK_DeletedNotification] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionHistoryMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionHistoryMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[DistributionId] [bigint] NULL,
	[AgniveerId] [bigint] NOT NULL,
	[Rank] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[TeamId] [bigint] NULL,
	[Location] [nvarchar](max) NULL,
	[UpdateCount] [bigint] NULL,
 CONSTRAINT [PK_DistributionHistoryMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[DograPercentage] [decimal](18, 2) NULL,
	[OICPercentage] [decimal](18, 2) NULL,
	[SikhPercentage] [decimal](18, 2) NULL,
 CONSTRAINT [PK_DistributionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[EquipmentMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[EquipmentMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[Category] [nvarchar](max) NULL,
	[Description] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Type] [nvarchar](max) NULL,
 CONSTRAINT [PK_EquipmentMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FilesMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FilesMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Files] [nvarchar](max) NULL,
	[Url] [nvarchar](max) NULL,
 CONSTRAINT [PK_FilesMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[LoginToken]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[LoginToken](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[UserName] [nvarchar](max) NOT NULL,
	[JwtToken] [nvarchar](max) NOT NULL,
	[RefreshToken] [nvarchar](max) NOT NULL,
	[ExpiryDate] [datetime2](7) NOT NULL,
	[CreatedDate] [datetime2](7) NOT NULL,
	[IsRevoked] [bit] NOT NULL,
 CONSTRAINT [PK_LoginToken] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[MedicalRecordMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[MedicalRecordMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[DoctorId] [bigint] NOT NULL,
	[Type] [nvarchar](max) NULL,
	[VisitDate] [datetime2](7) NULL,
	[FollowUpDate] [datetime2](7) NULL,
	[HospitalNameLocation] [nvarchar](max) NULL,
	[AdmitDate] [datetime2](7) NULL,
	[DischargeDate] [datetime2](7) NULL,
	[TreatmentGiven] [nvarchar](max) NULL,
	[Prescriptions] [nvarchar](max) NULL,
	[Diagnosis] [nvarchar](max) NULL,
	[BloodPressure] [nvarchar](max) NULL,
	[HeartRate] [nvarchar](max) NULL,
	[Weight] [decimal](18, 2) NULL,
	[Height] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[Status] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[LeaveType] [nvarchar](max) NULL,
	[ToDate] [datetime2](7) NULL,
 CONSTRAINT [PK_MedicalRecordMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCompanyHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCompanyHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCompanyHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[PlatoonNo] [nvarchar](max) NOT NULL,
	[PlatoonCommanderId] [bigint] NULL,
	[CompanyId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PoliceVerificationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PoliceVerificationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[SentDate] [datetime2](7) NULL,
	[ReceivedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PoliceVerificationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[RoleMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RoleMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Role] [nvarchar](max) NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_RoleMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSectionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSectionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionName] [nvarchar](max) NOT NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[IsExceptional] [bit] NULL,
 CONSTRAINT [PK_ScoreSectionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSubItemMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSubItemMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionId] [bigint] NULL,
	[Name] [nvarchar](max) NOT NULL,
	[MaxMarks] [decimal](18, 2) NULL,
	[Cutoff] [decimal](18, 2) NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_ScoreSubItemMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Email] [nvarchar](max) NULL,
	[ContactNo] [nvarchar](max) NULL,
	[Password] [nvarchar](max) NULL,
	[Username] [nvarchar](max) NULL,
	[FullName] [nvarchar](max) NULL,
	[AgniVeerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserRole]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserRole](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[RoleId] [bigint] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserRole] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_AgniveerId] ON [dbo].[AgniveerAttendanceMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_MarkedBy] ON [dbo].[AgniveerAttendanceMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_AgniveerId] ON [dbo].[AgniveerEquipment]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_EquipmentId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_EquipmentId] ON [dbo].[AgniveerEquipment]
(
	[EquipmentId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_AgniveerId] ON [dbo].[AgniveerLeaveMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_Id] ON [dbo].[AgniveerLeaveMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MarkedBy] ON [dbo].[AgniveerLeaveMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MedicalRecordId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MedicalRecordId] ON [dbo].[AgniveerLeaveMaster]
(
	[MedicalRecordId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [IX_AgniveerMaster_AgniveerNo]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerMaster_AgniveerNo] ON [dbo].[AgniveerMaster]
(
	[AgniveerNo] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_BatchId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_BatchId] ON [dbo].[AgniveerMaster]
(
	[BatchId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_PlatoonId] ON [dbo].[AgniveerMaster]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_SponserUnitId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_SponserUnitId] ON [dbo].[AgniveerMaster]
(
	[SponserUnitId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_AgniveerId] ON [dbo].[AgniveerPlatoonHistory]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_PlatoonId] ON [dbo].[AgniveerPlatoonHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_AgniveerId] ON [dbo].[AgniveerRelationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_DistributionId] ON [dbo].[AgniveerRelationMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_Id] ON [dbo].[AgniveerRelationMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_AgniveerId] ON [dbo].[AgniveerScoreAttempt]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_EvaluatedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_EvaluatedBy] ON [dbo].[AgniveerScoreAttempt]
(
	[EvaluatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_SubItemId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_SubItemId] ON [dbo].[AgniveerScoreAttempt]
(
	[SubItemId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_AgniveerId] ON [dbo].[AgniveerSectionResult]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_SectionId] ON [dbo].[AgniveerSectionResult]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CommanderId] ON [dbo].[CompanyCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CompanyId] ON [dbo].[CompanyCommanderHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CommandingOfficerId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CompanyId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CommandingOfficerId] ON [dbo].[CompanyMaster]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CompanyCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CompanyCommanderId] ON [dbo].[CompanyMaster]
(
	[CompanyCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_AgniveerId] ON [dbo].[DistributionHistoryMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_DistributionId] ON [dbo].[DistributionHistoryMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_DistributionMaster_Id] ON [dbo].[DistributionMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_AgniveerId] ON [dbo].[MedicalRecordMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_DoctorId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_DoctorId] ON [dbo].[MedicalRecordMaster]
(
	[DoctorId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_CommanderId] ON [dbo].[PlatoonCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_PlatoonId] ON [dbo].[PlatoonCommanderHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_CompanyId] ON [dbo].[PlatoonCompanyHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_PlatoonId] ON [dbo].[PlatoonCompanyHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_CompanyId] ON [dbo].[PlatoonMaster]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_PlatoonCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_PlatoonCommanderId] ON [dbo].[PlatoonMaster]
(
	[PlatoonCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PoliceVerificationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PoliceVerificationMaster_AgniveerId] ON [dbo].[PoliceVerificationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_ScoreSubItemMaster_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_ScoreSubItemMaster_SectionId] ON [dbo].[ScoreSubItemMaster]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserMaster_AgniVeerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserMaster_AgniVeerId] ON [dbo].[UserMaster]
(
	[AgniVeerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_RoleId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_RoleId] ON [dbo].[UserRole]
(
	[RoleId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_UserId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_UserId] ON [dbo].[UserRole]
(
	[UserId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnATTN'C']
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnEX PPG]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy] FOREIGN KEY([MarkedBy])
REFERENCES [dbo].[UserMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId] FOREIGN KEY([EquipmentId])
REFERENCES [dbo].[EquipmentMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId] FOREIGN KEY([MedicalRecordId])
REFERENCES [dbo].[MedicalRecordMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId]
GO
ALTER TABLE [dbo].
<truncated 401414 bytes>

NOTE: The output was truncated because it was too long. Use a more targeted query or a smaller range to get the information you need.
{{ CHECKPOINT 19 }}
 **The earlier parts of this conversation have been truncated due to its long length. The following content summarizes the truncated context so that you may continue your work. **


# User Requests
The following were user requests from the truncated conversation in chronological order:
1. i am talking about the logic 
2. let me know what types of things are there in database all the tables and there column names 
3. in detail what types things does that have in detailed 
4. user can ask these types of questions 
"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)
"""

QUESTION_BANK = \
{'by_category': {},
 'mixed': {'compare': ['Compare BPET and PPT scores.',
                       'Firing vs drill grading distribution.',
                       'BPET versus firing average marks.',
                       'Difference between PPT and drill performance.',
                       'Compare best attempts in BPET and PPT.',
                       'BPET vs drill improvement trends.',
                       'Compare top performers of Lakhwinder and Jaswant company in BPET.',
                       'Arora vs Thorat company attendan
<truncated 12284 bytes>
5. Which Agniveers who scored Excellent in BPET have completed police verification?

{
    "status": true,
    "message": "I couldn't understand the query clearly. Could you please rephrase it?",
    "formattedData": {
        "type": "MESSAGE",
        "title": "Conversation",
        "data": {
            "text": "I couldn't understand the query clearly. Could you please rephrase it?"
        }
    },
    "summary": "",
    "analysis": "",
    "prediction": "",
    "conclusion": "",
    "suggestedQuestions": [],
    "dotnetPayload": null,
    "sessionId": "TO6jvKjSthdVghjl58HyX"
}

top 10 BPET performers

{
    "status": true,
    "message": "The top 10 performers in BPET batch 1 are: AgniveerNo A0701749H Harminder Singh (784.00), AgniveerNo A0701557L Lakshay Katal (762.00), AgniveerNo A0701876X Tushar Chouhan (761.00), AgniveerNo A0702152M Kanwar Manhas (755.00), AgniveerNo A0702137X Manoj Kumar (751.00), AgniveerNo A0702198W Smilepreet Singh (750.00), AgniveerNo A0701602M Aditya Kanwar (750.00), AgniveerN
<truncated 3711 bytes>
6. PS E:\AgniAI> & E:\AgniAI\agniai-env\Scripts\Activate.ps1
(agniai-env) PS E:\AgniAI> python app.py
2026-07-17 17:29:41,133 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
2026-07-17 17:29:41,171 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/b207367332321f8e44f96e224ef15bc607f4dbf0/config_sentence_transformers.json "HTTP/1.1 200 OK"
2026-07-17 17:29:41,405 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-17 17:29:41,433 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/b207367332321f8e44f96e224ef15bc607f4dbf0/README.md "HTTP/1.1 200 OK"
2026-07-17 17:29:41,675 [INFO] httpx: HTTP Request: HEAD htt
<truncated 27380 bytes>
7. i would like to see the sql query inside dotnet payload in the response 
8. if it has by default gve n=10 remove that
9. You are a Senior SQL Server Architect, Database Reverse Engineer, .NET EF Core Expert, Query Optimization Specialist, and Enterprise Reporting Engineer.

You are responsible for converting an existing ASP.NET Core + Entity Framework backend into production-grade SQL Server queries that behave exactly like the backend.

This is NOT a SQL generation task.

This is a backend reverse engineering task.

Your objective is to understand every piece of business logic implemented inside the C# code and translate it into equivalent SQL.

PRIMARY OBJECTIVE

Whenever I provide C# backend files, Entity Framework LINQ, repositories, services, controllers, handlers, DTOs, models, or stored logic, you must:

Understand the business logic completely.
Ignore coding style.
Ignore variable names.
Ignore formatting.
Understand exactly how the backend calculates data.
Produce SQL that returns the exact same results.

The SQL must be functionally identical to the .NET implementation.

Never simplify business logic.

Never assume
<truncated 4119 bytes>
10. USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABAS
<truncated 448930 bytes>

# Previous Session Summary:
# 1. Outstanding User Requests
- **Task: Backend Reverse Engineering to SQL.** 
  - **Status:** IMPLEMENTATION.
  - **Detail:** Translate existing .NET business logic (LINQ/EF Core) into production-grade SQL Server queries. The user explicitly requested to think like EF Core, maintain exact business rules (including complex "Best Attempt" calculations), and provide the query name, purpose, backend method, logic breakdown, SQL query, and proof of equivalence.
- **Task: Fix SQL generator hallucination issues in cross-filter queries.**
  - **Status:** IMPLEMENTATION.
  - **Detail:** User previously requested to stop hallucinating columns/joins. Ongoing requirement to ensure the LLM references derived views correctly.

# 2. User Knowledge
- **Directives:**
  - "Understand the business logic completely. Ignore coding style/variable names/formatting."
  - "Never simplify business logic. Never assume. Never rewrite logic. Never optimize away business rules."
  - "Think like Entity Framework executing LINQ."
  - "If the code uses `IsBestAttempt`, determine what entity owns it, whether it is per sub-item/section/attempt/agniveer, and reproduce that exactly."
  - "Never assume 'Best Attempt'; you must inspect the code."
- **Preferences:**
  - Remove intent from logs; output generated SQL directly.
  - Inject SQL query into the `dotnetPayload` of the API response.
  - Remove default `n=10` limit for ranking operations.

# 3. Work Accomplished
- **Infrastructure:**
  - Updated `sql_executor.py` logging: Removed intent dictionary output, added explicit logging for generated SQL.
  - Modified `sql_executor.py`: Updated `_to_section` to accept and return the executed SQL query.
  - Modified `admin_pipeline.py`: Added logic to inject `sqlQuery` (or `sqlQueries` array) into `response_dotnet_payload` for SQL-backed requests and comparative datasets.
  - Removed `n=10` default limits: Updated `admin_intent.py` (parsing logic) and `sql_executor.py` (golden query renderer) to default to `SQL_MAX_ROWS` (500) instead of 10.
- **Pipeline:**
  - Refactored `sql_query_plan.py` to preserve original NL query structure for better generator accuracy.

# 4. Model Knowledge
- **Failures:** 
  - Previous attempts at SQL generation for cross-filter queries failed due to `ProgrammingError` ("could not be bound") where the LLM tried to access columns outside the JOIN scope.
- **Constraints:** SQL Server 2008 compatibility mode is strictly enforced. Derived views (`vw_*`) are the primary source of truth for calculations.
- **Architecture:** `admin_routes.py` -> `admin_pipeline.py` -> `sql_query_plan.py` -> `sql_executor.py`.

# 5. Files and Code
- **Edited Files:**
  - `e:\AgniAI\sql_executor.py`: (Lines ~640, ~760, ~850) Added SQL injection to responses, updated logging, removed `n=10` default.
  - `e:\AgniAI\admin_pipeline.py`: (Lines ~1260, ~1290) Injected `sqlQuery` into `dotnetPayload`.
  - `e:\AgniAI\intent_engine\admin_intent.py`: (Lines ~520-535) Removed `n=10` default logic for ranking.
- **Viewed Files:**
  - `e:\AgniAI\intent_engine\semantic_classifier.py`: Mapped `stage3` Ollama constrained logic.
  - All Table definitions from the provided DDL dump (used to map the join graph of the system).

# 6. Current Work and Next Steps
- **Next Steps:**
  1. Begin the "Backend Reverse Engineering" task requested by the user. 
  2. The user has provided the full database schema. I must now wait for the user to provide specific C# backend files (repositories/services) to map the logic.
  3. Continue refining the SQL generation to ensure it matches the exact business rules identified in the C# code.
- **User Quote:** "Whenever I provide C# backend files... you must: Understand the business logic completely... Produce SQL that returns the exact same results."

You have the 11 following artifacts written to the artifacts directory:

[ARTIFACT: database_schema]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/database_schema.md
Last Edited: 2026-07-17T07:19:34Z

[ARTIFACT: implementation_plan]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/implementation_plan.md
Last Edited: 2026-07-13T10:48:01Z

[ARTIFACT: debug_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/debug_query.py
Last Edited: 2026-07-13T11:08:49Z

[ARTIFACT: refactor]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/refactor.py
Last Edited: 2026-07-13T08:47:10Z

[ARTIFACT: run_views]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/run_views.py
Last Edited: 2026-07-17T07:55:58Z

[ARTIFACT: test_complex]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_complex.py
Last Edited: 2026-07-17T07:40:14Z

[ARTIFACT: test_db]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_db.py
Last Edited: 2026-07-13T09:21:47Z

[ARTIFACT: test_excellent]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_excellent.py
Last Edited: 2026-07-17T08:17:42Z

[ARTIFACT: test_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_query.py
Last Edited: 2026-07-13T11:08:26Z

[ARTIFACT: task]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/task.md
Last Edited: 2026-07-17T08:06:19Z

[ARTIFACT: walkthrough]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/walkthrough.md
Last Edited: 2026-07-17T08:06:43Z

# Conversation Logs

Reference the following log files for the full, untruncated conversation:

- C:\Users\dhruv\.gemini\antigravity-ide\brain\d6a094fd-1341-4992-9ec1-f64684561bbb\.system_generated\logs\transcript.jsonl

**IMPORTANT: this summary is just for your reference. You may respond to my previous and future messages, but DO NOT ACKNOWLEDGE THIS CHECKPOINT MESSAGE. JUST READ IT BUT DO NOT MENTION IT, RESPOND TO IT, OR TAKE ACTION BECAUSE OF IT.**
Created At: 2026-07-17T12:23:58Z
Completed At: 2026-07-17T12:23:58Z

				The command completed successfully.
				Output:
				<truncated 56 lines>
{"step_index":2409,"source":"USER_EXPLICIT","type":"USER_INPUT","status":"DONE","created_at":"2026-07-17T12:21:46Z","content":"<USER_REQUEST>\nUSE [master]\nGO\n/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/\nCREATE DATABASE [DB_Agni]\n CONTAINMENT = NONE\n ON  PRIMARY \n( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\\Program Files\\Microsoft SQL Server\\MSSQL16.SQLEXPRESS07\\MSSQL\\DATA\\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )\n LOG ON \n( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\\Program Files\\Microsoft SQL Server\\MSSQL16.SQLEXPRESS07\\MSSQL\\DATA\\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )\n WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF\nGO\nALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160\nGO\nIF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))\nbegin\nEXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'\nend\nGO\nALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF \nGO\nALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF \nGO\nALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF \nGO\nALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF \nGO\nALTER DATABASE [DB_Agni] SET ARITHABORT OFF \nGO\nALTER DATABASE [DB_Agni] SET AUTO_CLOSE OFF \nGO\nALTER DATABASE [DB_Agni] SET AUTO_SHRINK OFF \nGO\nALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS ON \nGO\nALTER DATABASE [DB_Agni] SET CURSOR_CLOSE_ON_COMMIT OFF \nGO\nALTER DATABASE [DB_Agni] SET CURSOR_DEFAULT  GLOBAL \nGO\nALTER DATABASE [DB_Agni] SET CONCAT_NULL_YIELDS_NULL OFF \nGO\nALTER DATABASE [DB_Agni] SET NUMERIC_ROUNDABORT OFF \nGO\nALTER DATABASE [DB_Agni] SET QUOTED_IDENTIFIER OFF \nGO\nALTER DATABASE [DB_Agni] SET RECURSIVE_TRIGGERS OFF \nGO\nALTER DATABASE [DB_Agni] SET  DISABLE_BROKER \nGO\nALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS_ASYNC OFF \nGO\nALTER DATABASE [DB_Agni] SET DATE_CORRELATION_OPTIMIZATION OFF \nGO\nALTER DATABASE [DB_Agni] SET TRUSTWORTHY OFF \nGO\nALTER DATABASE [DB_Agni] SET ALLOW_SNAPSHOT_ISOLATION OFF \nGO\nALTER DATABASE [DB_Agni] SET PARAMETERIZATION SIMPLE \nGO\nALTER DATABASE [DB_Agni] SET READ_COMMITTED_SNAPSHOT OFF \nGO\nALTER DATABASE [DB_Agni] SET HONOR_BROKER_PRIORITY OFF \nGO\nALTER DA\n<truncated 45216 bytes>\nit],(0))) FOR [OnEX PPG]\nGO\nALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])\nREFERENCES [dbo].[AgniveerMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId]\nGO\nALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy] FOREIGN KEY([MarkedBy])\nREFERENCES [dbo].[UserMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy]\nGO\nALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])\nREFERENCES [dbo].[AgniveerMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId]\nGO\nALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId] FOREIGN KEY([EquipmentId])\nREFERENCES [dbo].[EquipmentMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId]\nGO\nALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])\nREFERENCES [dbo].[AgniveerMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId]\nGO\nALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId] FOREIGN KEY([MedicalRecordId])\nREFERENCES [dbo].[MedicalRecordMaster] ([Id])\nGO\nALTER TABLE [dbo].[AgniveerLeaveMaster] CHECK CONSTRAINT [FK_AgniveerLeaveMaster_MedicalRecordMaster_MedicalRecordId]\nGO\nALTER TABLE [dbo].\n<truncated 401414 bytes>\n\nNOTE: The output was truncated because it was too long. Use a more targeted query or a smaller range to get the information you need.","truncated_fields":["content"]}
{"step_index":2413,"source":"USER_EXPLICIT","type":"USER_INPUT","status":"DONE","created_at":"2026-07-17T12:22:29Z","content":"<USER_REQUEST>\ni have given you the required files which are needed \n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nThe current local time is: 2026-07-17T17:52:29+05:30.\n\nThe user's current state is as follows:\nActive Document: e:\\AgniAI\\sql_executor.py (LANGUAGE_PYTHON)\nCursor is on line: 882\nOther open documents:\n- e:\\AgniAI\\sql_executor.py (LANGUAGE_PYTHON)\n- e:\\AgniAI\\intent_engine\\admin_intent.py (LANGUAGE_PYTHON)\n- e:\\AgniAI\\admin_pipeline.py (LANGUAGE_PYTHON)\nRunning terminal commands:\n- .\\ngrok.exe http --domain=attest-enroll-handsfree.ngrok-free.dev 5000 (in e:\\AgniAI, running for 23m37s)\n- python app.py (in e:\\AgniAI, running for 12m57s)\n</ADDITIONAL_METADATA>"}
{"step_index":2433,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","created_at":"2026-07-17T12:23:35Z","thinking":"**Prioritizing Specific Tools**\n\nI'm focused on utilizing the specialized tools I have, like `view_file` and `replace_file_content`, instead of falling back on more generic commands. My goal is to leverage these tools to their fullest potential and avoid any unnecessary redundancy in my approach. This focused strategy should make everything much more efficient.\n\n\n**Analyzing Database Schema**\n\nI'm now analyzing the implications of the provided database schema. It seems the user gave me the schema in the belief that they gave me \"the required files\". I am considering whether the \"FIRST TASK\" can be performed by looking *only* at the schema; or whether I need C# backend files to perform the analysis as specified in the original instructions. It's a key question of how to proceed with the analysis.\n\n\n","tool_calls":[{"name":"grep_search","args":{"CaseInsensitive":"false","IsRegex":"false","MatchPerLine":"true","Query":"\"\\\"type\\\":\\\"USER_INPUT\\\"\"","SearchPath":"\"C:\\\\Users\\\\dhruv\\\\.gemini\\\\antigravity-ide\\\\brain\\\\d6a094fd-1341-4992-9ec1-f64684561bbb\\\\.system_generated\\\\logs\\\\transcript.jsonl\""}}]}


<USER_REQUEST>
Search the schema.

Does the following actually exist?

vw_AgniveerBestAttemptTotals

vw_AgniveerSectionGrades

vw_AgniveerLeaveDayCounts

vw_AgniveerLeaveThreshold

vw_AgniveerAttendanceStatus

vw_EquipmentDegraded

vw_AgniveerVerificationStatus

vw_AgniveerBmi

For each view determine

Exists?

Missing?

Different name?

Different columns?

Different logic?

If missing

DO NOT simply assume it exists.

Instead choose ONE of the following


this is the database schema 
USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [DB_Agni] SET ARITHABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [DB_Agni] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [DB_Agni] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [DB_Agni] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [DB_Agni] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [DB_Agni] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [DB_Agni] SET  DISABLE_BROKER 
GO
ALTER DATABASE [DB_Agni] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [DB_Agni] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [DB_Agni] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [DB_Agni] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [DB_Agni] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [DB_Agni] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [DB_Agni] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [DB_Agni] SET  RESTRICTED_USER 
GO
ALTER DATABASE [DB_Agni] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [DB_Agni] SET DB_CHAINING OFF 
GO
ALTER DATABASE [DB_Agni] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [DB_Agni] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [DB_Agni] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [DB_Agni] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE = ON
GO
ALTER DATABASE [DB_Agni] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [DB_Agni]
GO
/****** Object:  Table [dbo].[__EFMigrationsHistory]    Script Date: 17/07/2026 11:05:21 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[__EFMigrationsHistory](
	[MigrationId] [nvarchar](150) NOT NULL,
	[ProductVersion] [nvarchar](32) NOT NULL,
 CONSTRAINT [PK___EFMigrationsHistory] PRIMARY KEY CLUSTERED 
(
	[MigrationId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerAttendanceMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerAttendanceMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[AttendanceDateTime] [datetime2](7) NOT NULL,
	[IsPresent] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerAttendanceMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerEquipment]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerEquipment](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[EquipmentId] [bigint] NULL,
	[Type] [nvarchar](max) NULL,
	[GivenDateTime] [datetime2](7) NULL,
	[ReturnDateTime] [datetime2](7) NULL,
	[GivenCondition] [nvarchar](max) NULL,
	[ReturnCondition] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerEquipment] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerLeaveMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerLeaveMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[OnAnnualLeave] [bit] NOT NULL,
	[OnMedicalLeave] [bit] NOT NULL,
	[OnSickLeave] [bit] NOT NULL,
	[IsHospitalized] [bit] NOT NULL,
	[IsAbscondedLeave] [bit] NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[MarkedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[ToDate] [datetime2](7) NULL,
	[OnATTN'C'] [bit] NOT NULL,
	[OnEX PPG] [bit] NOT NULL,
	[MedicalRecordId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerLeaveMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[FullName] [nvarchar](max) NOT NULL,
	[AgniveerNo] [nvarchar](450) NOT NULL,
	[DateOfBirth] [datetime2](7) NULL,
	[DateOfJoining] [datetime2](7) NULL,
	[Address] [nvarchar](max) NULL,
	[MobileNo] [nvarchar](max) NULL,
	[EroName] [nvarchar](max) NULL,
	[NextOfKin] [nvarchar](max) NULL,
	[Class] [nvarchar](max) NULL,
	[BloodGroup] [nvarchar](max) NULL,
	[Height] [decimal](18, 2) NULL,
	[Weight] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[PlatoonId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[PhotoPath] [nvarchar](max) NULL,
	[District] [nvarchar](max) NULL,
	[Email] [nvarchar](max) NULL,
	[EnrolledId] [nvarchar](max) NULL,
	[HouseNo] [nvarchar](max) NULL,
	[IdMarkI] [nvarchar](max) NULL,
	[IdMarkI1] [nvarchar](max) NULL,
	[MainCategory] [nvarchar](max) NULL,
	[PinCode] [nvarchar](max) NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[PostOffice] [nvarchar](max) NULL,
	[Qualification] [nvarchar](max) NULL,
	[State] [nvarchar](max) NULL,
	[Tehsil] [nvarchar](max) NULL,
	[Village] [nvarchar](max) NULL,
	[BatchId] [bigint] NULL,
	[Awards] [nvarchar](max) NULL,
	[Certificate] [nvarchar](max) NULL,
	[Hobby] [nvarchar](max) NULL,
	[Skill] [nvarchar](max) NULL,
	[Sports] [nvarchar](max) NULL,
	[IsDisqualified] [bit] NULL,
	[Remarks] [nvarchar](max) NULL,
	[SponserUnitId] [bigint] NULL,
	[DisqualifiedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerPlatoonHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerPlatoonHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerPlatoonHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerRelationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerRelationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[RelationName] [nvarchar](max) NULL,
	[RelationType] [nvarchar](max) NULL,
	[DistributionId] [bigint] NULL,
 CONSTRAINT [PK_AgniveerRelationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerScoreAttempt]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerScoreAttempt](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SubItemId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[MarksObtained] [decimal](18, 2) NULL,
	[IsBestAttempt] [bit] NULL,
	[AttemptedDate] [datetime2](7) NULL,
	[EvaluatedBy] [bigint] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_AgniveerScoreAttempt] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AgniveerSectionResult]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AgniveerSectionResult](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[SectionId] [bigint] NULL,
	[AttemptNo] [nvarchar](max) NULL,
	[OmrInputTotal] [decimal](18, 2) NULL,
	[Grading] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[SubItemTotalMarks] [decimal](18, 2) NULL,
	[ExceptionalMarks] [decimal](18, 2) NULL,
 CONSTRAINT [PK_AgniveerSectionResult] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BatchMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BatchMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[BatchName] [nvarchar](max) NOT NULL,
	[StartDate] [datetime2](7) NULL,
	[TrainingEndDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NULL,
	[ServiceEndDate] [datetime2](7) NULL,
	[DisplayOrder] [bigint] NULL,
 CONSTRAINT [PK_BatchMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyCommandingOfficerHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyCommandingOfficerHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[CommandingOfficerId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyCommandingOfficerHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanyMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanyMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[CompanyCommanderId] [bigint] NULL,
	[CommandingOfficerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_CompanyMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CompanySchedule]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CompanySchedule](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Pd] [nvarchar](max) NULL,
	[TimeRange] [nvarchar](max) NULL,
	[Code] [nvarchar](max) NULL,
	[Type] [nvarchar](max) NULL,
	[Details] [nvarchar](max) NULL,
	[Location] [nvarchar](max) NULL,
	[Resp] [nvarchar](max) NULL,
	[ScheduleDate] [datetime2](7) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[CompanyId] [bigint] NULL,
 CONSTRAINT [PK_CompanySchedule] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DeletedNotification]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DeletedNotification](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NULL,
	[IssueType] [nvarchar](max) NULL,
 CONSTRAINT [PK_DeletedNotification] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionHistoryMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionHistoryMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[DistributionId] [bigint] NULL,
	[AgniveerId] [bigint] NOT NULL,
	[Rank] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[TeamId] [bigint] NULL,
	[Location] [nvarchar](max) NULL,
	[UpdateCount] [bigint] NULL,
 CONSTRAINT [PK_DistributionHistoryMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DistributionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DistributionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[DograPercentage] [decimal](18, 2) NULL,
	[OICPercentage] [decimal](18, 2) NULL,
	[SikhPercentage] [decimal](18, 2) NULL,
 CONSTRAINT [PK_DistributionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[EquipmentMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[EquipmentMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[Category] [nvarchar](max) NULL,
	[Description] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[Type] [nvarchar](max) NULL,
 CONSTRAINT [PK_EquipmentMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FilesMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FilesMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Files] [nvarchar](max) NULL,
	[Url] [nvarchar](max) NULL,
 CONSTRAINT [PK_FilesMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[LoginToken]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[LoginToken](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[UserName] [nvarchar](max) NOT NULL,
	[JwtToken] [nvarchar](max) NOT NULL,
	[RefreshToken] [nvarchar](max) NOT NULL,
	[ExpiryDate] [datetime2](7) NOT NULL,
	[CreatedDate] [datetime2](7) NOT NULL,
	[IsRevoked] [bit] NOT NULL,
 CONSTRAINT [PK_LoginToken] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[MedicalRecordMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[MedicalRecordMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[DoctorId] [bigint] NOT NULL,
	[Type] [nvarchar](max) NULL,
	[VisitDate] [datetime2](7) NULL,
	[FollowUpDate] [datetime2](7) NULL,
	[HospitalNameLocation] [nvarchar](max) NULL,
	[AdmitDate] [datetime2](7) NULL,
	[DischargeDate] [datetime2](7) NULL,
	[TreatmentGiven] [nvarchar](max) NULL,
	[Prescriptions] [nvarchar](max) NULL,
	[Diagnosis] [nvarchar](max) NULL,
	[BloodPressure] [nvarchar](max) NULL,
	[HeartRate] [nvarchar](max) NULL,
	[Weight] [decimal](18, 2) NULL,
	[Height] [decimal](18, 2) NULL,
	[EyeSight] [decimal](18, 2) NULL,
	[Status] [nvarchar](max) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[FromDate] [datetime2](7) NULL,
	[LeaveType] [nvarchar](max) NULL,
	[ToDate] [datetime2](7) NULL,
 CONSTRAINT [PK_MedicalRecordMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCommanderHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCommanderHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CommanderId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCommanderHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonCompanyHistory]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonCompanyHistory](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[PlatoonId] [bigint] NOT NULL,
	[CompanyId] [bigint] NOT NULL,
	[StartDate] [datetime2](7) NOT NULL,
	[EndDate] [datetime2](7) NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonCompanyHistory] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PlatoonMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PlatoonMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Name] [nvarchar](max) NOT NULL,
	[PlatoonNo] [nvarchar](max) NOT NULL,
	[PlatoonCommanderId] [bigint] NULL,
	[CompanyId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PlatoonMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PoliceVerificationMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PoliceVerificationMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[AgniveerId] [bigint] NOT NULL,
	[PoliceStation] [nvarchar](max) NULL,
	[SentDate] [datetime2](7) NULL,
	[ReceivedDate] [datetime2](7) NULL,
	[Status] [nvarchar](max) NOT NULL,
	[Remarks] [nvarchar](max) NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_PoliceVerificationMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[RoleMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RoleMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Role] [nvarchar](max) NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_RoleMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSectionMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSectionMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionName] [nvarchar](max) NOT NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
	[IsExceptional] [bit] NULL,
 CONSTRAINT [PK_ScoreSectionMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ScoreSubItemMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ScoreSubItemMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[SectionId] [bigint] NULL,
	[Name] [nvarchar](max) NOT NULL,
	[MaxMarks] [decimal](18, 2) NULL,
	[Cutoff] [decimal](18, 2) NULL,
	[DisplayOrder] [int] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_ScoreSubItemMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserMaster]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserMaster](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[Email] [nvarchar](max) NULL,
	[ContactNo] [nvarchar](max) NULL,
	[Password] [nvarchar](max) NULL,
	[Username] [nvarchar](max) NULL,
	[FullName] [nvarchar](max) NULL,
	[AgniVeerId] [bigint] NULL,
	[IsActive] [bit] NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserMaster] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[UserRole]    Script Date: 17/07/2026 11:05:22 AM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[UserRole](
	[Id] [bigint] IDENTITY(1,1) NOT NULL,
	[UserId] [bigint] NOT NULL,
	[RoleId] [bigint] NOT NULL,
	[InsertedBy] [bigint] NULL,
	[InsertedDate] [datetime2](7) NOT NULL,
	[UpdatedBy] [bigint] NULL,
	[UpdatedDate] [datetime2](7) NULL,
 CONSTRAINT [PK_UserRole] PRIMARY KEY CLUSTERED 
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_AgniveerId] ON [dbo].[AgniveerAttendanceMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerAttendanceMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerAttendanceMaster_MarkedBy] ON [dbo].[AgniveerAttendanceMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_AgniveerId] ON [dbo].[AgniveerEquipment]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerEquipment_EquipmentId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerEquipment_EquipmentId] ON [dbo].[AgniveerEquipment]
(
	[EquipmentId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_AgniveerId] ON [dbo].[AgniveerLeaveMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_Id] ON [dbo].[AgniveerLeaveMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MarkedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MarkedBy] ON [dbo].[AgniveerLeaveMaster]
(
	[MarkedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerLeaveMaster_MedicalRecordId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerLeaveMaster_MedicalRecordId] ON [dbo].[AgniveerLeaveMaster]
(
	[MedicalRecordId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [IX_AgniveerMaster_AgniveerNo]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerMaster_AgniveerNo] ON [dbo].[AgniveerMaster]
(
	[AgniveerNo] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_BatchId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_BatchId] ON [dbo].[AgniveerMaster]
(
	[BatchId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_PlatoonId] ON [dbo].[AgniveerMaster]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerMaster_SponserUnitId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerMaster_SponserUnitId] ON [dbo].[AgniveerMaster]
(
	[SponserUnitId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_AgniveerId] ON [dbo].[AgniveerPlatoonHistory]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerPlatoonHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerPlatoonHistory_PlatoonId] ON [dbo].[AgniveerPlatoonHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_AgniveerId] ON [dbo].[AgniveerRelationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_DistributionId] ON [dbo].[AgniveerRelationMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerRelationMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_AgniveerRelationMaster_Id] ON [dbo].[AgniveerRelationMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_AgniveerId] ON [dbo].[AgniveerScoreAttempt]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_EvaluatedBy]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_EvaluatedBy] ON [dbo].[AgniveerScoreAttempt]
(
	[EvaluatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerScoreAttempt_SubItemId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerScoreAttempt_SubItemId] ON [dbo].[AgniveerScoreAttempt]
(
	[SubItemId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_AgniveerId] ON [dbo].[AgniveerSectionResult]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_AgniveerSectionResult_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_AgniveerSectionResult_SectionId] ON [dbo].[AgniveerSectionResult]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CommanderId] ON [dbo].[CompanyCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommanderHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommanderHistory_CompanyId] ON [dbo].[CompanyCommanderHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CommandingOfficerId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyCommandingOfficerHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyCommandingOfficerHistory_CompanyId] ON [dbo].[CompanyCommandingOfficerHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CommandingOfficerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CommandingOfficerId] ON [dbo].[CompanyMaster]
(
	[CommandingOfficerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_CompanyMaster_CompanyCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_CompanyMaster_CompanyCommanderId] ON [dbo].[CompanyMaster]
(
	[CompanyCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_AgniveerId] ON [dbo].[DistributionHistoryMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionHistoryMaster_DistributionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_DistributionHistoryMaster_DistributionId] ON [dbo].[DistributionHistoryMaster]
(
	[DistributionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_DistributionMaster_Id]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE UNIQUE NONCLUSTERED INDEX [IX_DistributionMaster_Id] ON [dbo].[DistributionMaster]
(
	[Id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_AgniveerId] ON [dbo].[MedicalRecordMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_MedicalRecordMaster_DoctorId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_MedicalRecordMaster_DoctorId] ON [dbo].[MedicalRecordMaster]
(
	[DoctorId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_CommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_CommanderId] ON [dbo].[PlatoonCommanderHistory]
(
	[CommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCommanderHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCommanderHistory_PlatoonId] ON [dbo].[PlatoonCommanderHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_CompanyId] ON [dbo].[PlatoonCompanyHistory]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonCompanyHistory_PlatoonId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonCompanyHistory_PlatoonId] ON [dbo].[PlatoonCompanyHistory]
(
	[PlatoonId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_CompanyId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_CompanyId] ON [dbo].[PlatoonMaster]
(
	[CompanyId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PlatoonMaster_PlatoonCommanderId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PlatoonMaster_PlatoonCommanderId] ON [dbo].[PlatoonMaster]
(
	[PlatoonCommanderId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_PoliceVerificationMaster_AgniveerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_PoliceVerificationMaster_AgniveerId] ON [dbo].[PoliceVerificationMaster]
(
	[AgniveerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_ScoreSubItemMaster_SectionId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_ScoreSubItemMaster_SectionId] ON [dbo].[ScoreSubItemMaster]
(
	[SectionId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserMaster_AgniVeerId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserMaster_AgniVeerId] ON [dbo].[UserMaster]
(
	[AgniVeerId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_RoleId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_RoleId] ON [dbo].[UserRole]
(
	[RoleId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IX_UserRole_UserId]    Script Date: 17/07/2026 11:05:22 AM ******/
CREATE NONCLUSTERED INDEX [IX_UserRole_UserId] ON [dbo].[UserRole]
(
	[UserId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnATTN'C']
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster] ADD  DEFAULT (CONVERT([bit],(0))) FOR [OnEX PPG]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy] FOREIGN KEY([MarkedBy])
REFERENCES [dbo].[UserMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerAttendanceMaster] CHECK CONSTRAINT [FK_AgniveerAttendanceMaster_UserMaster_MarkedBy]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_AgniveerMaster_AgniveerId]
GO
ALTER TABLE [dbo].[AgniveerEquipment]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId] FOREIGN KEY([EquipmentId])
REFERENCES [dbo].[EquipmentMaster] ([Id])
GO
ALTER TABLE [dbo].[AgniveerEquipment] CHECK CONSTRAINT [FK_AgniveerEquipment_EquipmentMaster_EquipmentId]
GO
ALTER TABLE [dbo].[AgniveerLeaveMaster]  WITH CHECK ADD  CONSTRAINT [FK_AgniveerLeaveMaster_AgniveerMaster_AgniveerId] FOREIGN KEY([AgniveerId])
REFERENCES [dbo].[AgniveerMa
<truncated 11702 bytes>

NOTE: The output was truncated because it was too long. Use a more targeted query or a smaller range to get the information you need.
{{ CHECKPOINT 21 }}
 **The earlier parts of this conversation have been truncated due to its long length. The following content summarizes the truncated context so that you may continue your work. **


# User Requests
The following were user requests from the truncated conversation in chronological order:
1. if it has by default gve n=10 remove that
2. You are a Senior SQL Server Architect, Database Reverse Engineer, .NET EF Core Expert, Query Optimization Specialist, and Enterprise Reporting Engineer.

You are responsible for converting an existing ASP.NET Core + Entity Framework backend into production-grade SQL Server queries that behave exactly like the backend.

This is NOT a SQL generation task.

This is a backend reverse engineering task.

Your objective is to understand every piece of business logic implemented inside the C# code and translate it into equivalent SQL.

PRIMARY OBJECTIVE

Whenever I provide C# backend files, Entity Framework LINQ, repositories, services, controllers, handlers, DTOs, models, or stored logic, you must:

Understand the business logic completely.
Ignore coding style.
Ignore variable names.
Ignore formatting.
Understand exactly how the backend calculates data.
Produce SQL that returns the exact same results.

The SQL must be functionally identical to the .NET implementation.

Never simplify business logic.

Never assume
<truncated 4119 bytes>
3. USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [DB_Agni] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [DB_Agni].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [DB_Agni] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE
<truncated 448929 bytes>
4. i have given you the required files which are needed 
5. you have to inspect my code and solve this 
6. # SUPER PROMPT — Enterprise Database Audit & SQL Executor Refactoring (Production Grade)
You are a **Principal Database Architect**, **Senior SQL Server Engineer**, **Enterprise AI Architect**, **LLM Text-to-SQL Specialist**, and **Code Auditor**.

You are auditing and refactoring an Enterprise AI SQL execution engine that converts natural language into SQL queries.

This system **must always return exactly the same results as the original .NET backend** while allowing LLM-generated SQL for questions that are not already covered by Golden Queries.

Your objective is **NOT** merely to fix syntax.

Your objective is to make the SQL execution layer **100% consistent with the actual database**, eliminate assumptions, prevent hallucinations, and guarantee correctness.

---

# IMPORTANT

The following files are the ONLY source of truth.

1. sql_executor.py
2. Complete SQL Server Database Schema

Never trust comments.

Never trust manually written schema documentation.

Never trust assu
<truncated 10167 bytes>
7. You are a Senior SQL Server Architect, Database Reverse Engineer, .NET EF Core Expert, Enterprise Reporting Engineer, and Leave Management Domain Expert.

Your responsibility is to reverse engineer the Leave module of the ASP.NET Core backend and produce SQL Server queries that return exactly the same results as the Entity Framework implementation.

This is not SQL generation.

This is backend logic replication.

Your SQL must behave exactly like the C# backend.

PRIMARY OBJECTIVE

Given the backend source code (Controllers, Services, Entity Framework LINQ, Helpers, DTOs, Domain Handlers, and Models), reverse engineer every Leave operation and convert it into production-ready SQL.

Never simplify the logic.

Never assume business rules.

Never replace backend calculations with your own implementation.

If a helper method already implements business logic, the SQL must reproduce that helper.

THINK LIKE THE BACKEND

Do NOT think like ChatGPT.

Think like Entity Framework executing LINQ.

Discover:

Primary 
<truncated 5013 bytes>
8. ROLE

You are a Principal Database Architect, Senior SQL Server Engineer, Enterprise AI Architect, LLM Text-to-SQL Specialist, and Code Auditor.

You are auditing and refactoring an Enterprise AI SQL execution engine that converts natural language into SQL queries.

This system must always return exactly the same results as the original .NET backend while allowing LLM-generated SQL for questions that are not already covered by Golden Queries.

Your objective is NOT merely to fix syntax.

Your objective is to make the SQL execution layer 100% consistent with the actual database, eliminate assumptions, prevent hallucinations, and guarantee correctness.

IMPORTANT

The following files are the ONLY source of truth.

sql_executor.py
Complete SQL Server Database Schema

Never trust comments.

Never trust manually written schema documentation.

Never trust assumptions.

Only trust what actually exists inside the database schema.

PRIMARY OBJECTIVES

Audit the entire SQL execution engine against the database schem
<truncated 8961 bytes>
9. does view exist in database
10. Search the schema.

Does the following actually exist?

vw_AgniveerBestAttemptTotals

vw_AgniveerSectionGrades

vw_AgniveerLeaveDayCounts

vw_AgniveerLeaveThreshold

vw_AgniveerAttendanceStatus

vw_EquipmentDegraded

vw_AgniveerVerificationStatus

vw_AgniveerBmi

For each view determine

Exists?

Missing?

Different name?

Different columns?

Different logic?

If missing

DO NOT simply assume it exists.

Instead choose ONE of the following


this is the database schema 
USE [master]
GO
/****** Object:  Database [DB_Agni]    Script Date: 17/07/2026 11:05:21 AM ******/
CREATE DATABASE [DB_Agni]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'DB_AgniveerAITMS', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'DB_AgniveerAITMS_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS07\MSSQL\DATA\DB_Agni_log.ldf' , SIZE = 204800KB , MAXSIZE = 2048GB , FILEGROWTH = 65
<truncated 59095 bytes>

# Previous Session Summary:
# 1. Outstanding User Requests
- **Task: Leave Module Reverse Engineering & Replication.**
  - **Status:** PLANNING / DISCOVERY.
  - **Detail:** Reverse engineer all leave-related operations (Most Leave, Least Leave, Current Leave, Absconded) from the .NET backend into exact SQL Server equivalents. The user requires replication of business logic, DTO shapes, and filters, using derived views where available.
- **Task: Schema Audit & Alignment.**
  - **Status:** IMPLEMENTATION.
  - **Detail:** Verify the existence of specific views (`vw_AgniveerBestAttemptTotals`, `vw_AgniveerLeaveDayCounts`, etc.) and rectify discrepancies between assumed views and actual database content.

# 2. User Knowledge
- **Directives:**
  - "Never simplify the logic. Never assume business rules. Never replace backend calculations with your own implementation."
  - "Think like Entity Framework executing LINQ."
  - "If any rule cannot be proven from the code, explicitly state: 'Cannot infer this behavior from the provided backend.'"
- **Context:**
  - Provided DDL script for `DB_Agni`.
  - Provided `.env` file containing connection string details (using `ODBC Driver 18 for SQL Server`).
  - User is monitoring progress via `app.py` and `ngrok`.

# 3. Work Accomplished
- **Infrastructure:**
  - Successfully refactored `sql_executor.py` to remove static `SCHEMA_CARD` and implement dynamic schema discovery via `sql_schema_guard.py`.
  - Implemented centralized `business_rules.py`.
  - Implemented read-only SQL validation gate (Step 19).
- **Audit:**
  - Identified that the requested views (e.g., `vw_AgniveerLeaveDayCounts`) were assumed to exist in the LLM prompt context but are not explicitly visible in the DDL provided.
  - Verified connection string configuration via `.env` inspection.

# 4. Model Knowledge
- **Architecture:** The system now uses a dynamic generation pipeline: `sql_schema_guard` probes `INFORMATION_SCHEMA` -> `sql_executor` constructs prompt with `business_rules.py` -> LLM generates SQL -> Validation Gate -> Execute.
- **Failures:** 
  - `pyodbc` connection attempts failed due to driver/environment issues in the test container (`IM002` error). The next agent should focus on static DDL inspection rather than live `INFORMATION_SCHEMA` polling if the driver remains unavailable.
- **Constraints:** SQL Server 2008 compatibility mode must be maintained.

# 5. Files and Code
- **Edited Files:**
  - `e:\AgniAI\sql_executor.py`: (Throughout) Removed `SCHEMA_CARD`, integrated dynamic schema/business rules, added safety validation.
  - `e:\AgniAI\business_rules.py`: (Created/Updated) Centralized constants and `LLM_HARD_RULES`.
  - `e:\AgniAI\task.md`: (Updated) Tracked progress.
- **Viewed Files:**
  - `e:\AgniAI\.env`: Inspected for `SQL_READONLY_CONN` and connectivity settings.
  - `e:\AgniAI\sql_schema_guard.py`: Inspected for schema discovery implementation.

# 6. Current Work and Next Steps
- **Current Work:** The user is asking to verify the existence of specific views (`vw_*`). Since the DDL provided does not contain `CREATE VIEW` statements, I must report their absence and recommend whether to build them or replace them with equivalent raw SQL queries as per the "Step 4" guidelines.
- **Next Steps:**
  1. Formally report to the user that the requested `vw_*` views are **missing** from the provided DDL.
  2. Propose a plan: Either provide the `CREATE VIEW` definitions for me to include, or (preferred) rewrite the Golden Queries to use raw table logic that replicates the intended view behavior exactly as the .NET backend would.
  3. Begin the requested Leave Module reverse engineering using the raw tables identified in the DDL (`AgniveerLeaveMaster`, `AgniveerMaster`, `MedicalRecordMaster`).

You have the 11 following artifacts written to the artifacts directory:

[ARTIFACT: database_schema]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/database_schema.md
Last Edited: 2026-07-17T07:19:34Z

[ARTIFACT: implementation_plan]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/implementation_plan.md
Last Edited: 2026-07-17T12:29:04Z

[ARTIFACT: debug_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/debug_query.py
Last Edited: 2026-07-13T11:08:49Z

[ARTIFACT: refactor]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/refactor.py
Last Edited: 2026-07-13T08:47:10Z

[ARTIFACT: run_views]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/run_views.py
Last Edited: 2026-07-17T07:55:58Z

[ARTIFACT: test_complex]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_complex.py
Last Edited: 2026-07-17T07:40:14Z

[ARTIFACT: test_db]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_db.py
Last Edited: 2026-07-13T09:21:47Z

[ARTIFACT: test_excellent]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_excellent.py
Last Edited: 2026-07-17T08:17:42Z

[ARTIFACT: test_query]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/scratch/test_query.py
Last Edited: 2026-07-13T11:08:26Z

[ARTIFACT: task]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/task.md
Last Edited: 2026-07-17T12:35:56Z

[ARTIFACT: walkthrough]
Path: file:///C:/Users/dhruv/.gemini/antigravity-ide/brain/d6a094fd-1341-4992-9ec1-f64684561bbb/walkthrough.md
Last Edited: 2026-07-17T12:35:41Z

# Conversation Logs

Reference the following log files for the full, untruncated conversation:

- C:\Users\dhruv\.gemini\antigravity-ide\brain\d6a094fd-1341-4992-9ec1-f64684561bbb\.system_generated\logs\transcript.jsonl

**IMPORTANT: this summary is just for your reference. You may respond to my previous and future messages, but DO NOT ACKNOWLEDGE THIS CHECKPOINT MESSAGE. JUST READ IT BUT DO NOT MENTION IT, RESPOND TO IT, OR TAKE ACTION BECAUSE OF IT.**
Created At: 2026-07-17T12:55:45Z
Completed At: 2026-07-17T12:55:45Z

				The command failed with exit code: 1
				Output:
				Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import json; [print(json.loads(line)['content']) for line in open(r'C:\Users\dhruv\.gemini\antigravity-ide\brain\d6a094fd-1341-4992-9ec1-f64684561bbb\.system_generated\logs\transcript_full.jsonl') if 'CREATE DATABASE [DB_Agni]' in json.loads(line).get('content', '')]
                                                                 ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\dhruv\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 23, in decode
    return codecs.charmap_decode(input,self.errors,decoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1374: character maps to <undefined>


